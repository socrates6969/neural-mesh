// bench/ruflo_train2.mjs - DESCRIPTIVE, NOT GATED (R122): train the parts of ruflo that `hooks route` / `hooks_route` ACTUALLY READ, from the
// v2 structural labels, and measure held-out route accuracy before/after on a chronological split (first 70% train, last 30% test).
//
// What `hooks_route` reads (@claude-flow/cli dist/src/mcp-tools/hooks-tools.js, hooksRoute, 3.56.x), in order:
//   1. AgentDB bridge (bridgeRouteTask): hard-disabled on win32 (plugin-medic) -> never answers here.
//   2. SONA persisted patterns (memory/sona-optimizer.js getRoutingSuggestion -> source 'sona-pattern'), used only when confidence >= 0.6 AND
//      the agent is in AGENT_TYPES (coder, tester, reviewer, architect, researcher, optimizer, debugger, documenter, security-architect,
//      performance-engineer) - an fr-* name is NEVER eligible on this path.  Written by hooks_intelligence_trajectory-end (processTrajectoryOutcome):
//      a pattern is keyed by the task's first <=10 significant keywords + agent, confidence 0.5 +0.1*(1-c) per success; a route needs overlap*confidence >= 0.6.
//   3. routeTaskLocal: a semantic index (default embedder 'hash' here) over static TASK_PATTERNS + `learned-<agent>` patterns rebuilt from
//      .claude-flow/routing-outcomes.json (written by hooks_post-task with task+agent+success+quality). A match needs similarity > 0.4 (query vs ONE
//      keyword), and a learned pattern needs support >= 2 and reliability >= 0.75. Otherwise keyword fallback -> 'coder' default.
// The previous bench (ruflo_train.mjs) trained post-task with the ruflo role names via the CLI and `hooks intelligence --train` (which feeds neither store).
//
// This script calls the tool handlers in-process (the same code the MCP server runs) with cwd = a fresh sandbox, so ruflo's .claude-flow/.swarm state never
// touches the repo and no database file is edited by hand. Variants:
//   base            : no training (control)
//   postask_native  : hooks_post-task per train row with the NATIVE fr-* agent name; success=true/quality .9 only for outcome 'passed', else success=false/.4/.2
//   postask_imit    : same, but every row counts as a success (imitation; isolates the label-quality effect)
//   postask_ruflo   : as postask_native but with ruflo role names (coder/tester/...) mapped back for scoring (the old bench's convention)
//   sona_traj       : hooks_intelligence_trajectory-start/step/end per train row with ruflo role names (the only names the SONA path accepts), v2 success
//   all             : postask_native + sona_traj
// Run:  RUFLO_PKG_SRC=<.../@claude-flow/cli/dist/src> node bench/ruflo_train2.mjs --variant <v> [--data bench/data/ruflo_dispatch_v2.jsonl] [--out FILE]
//       node bench/ruflo_train2.mjs --all   (spawns every variant in parallel, merges into bench/ruflo_train2_result.json)
import fs from 'node:fs'; import path from 'node:path'; import os from 'node:os'; import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROLES = ['fr-planner','fr-builder','fr-tester','fr-reviewer','fr-scriber','fr-shipper','fr-ci-diagnoser','fr-security-reviewer'];
const TO_RUFLO = { 'fr-planner':'architect','fr-builder':'coder','fr-tester':'tester','fr-reviewer':'reviewer','fr-scriber':'documenter','fr-shipper':'optimizer','fr-ci-diagnoser':'debugger','fr-security-reviewer':'security-architect' };
const FROM_RUFLO = Object.fromEntries(Object.entries(TO_RUFLO).map(([k, v]) => [v, k]));
Object.assign(FROM_RUFLO, { planner:'fr-planner', 'backend-dev':'fr-builder', 'frontend-dev':'fr-builder', 'performance-engineer':'fr-builder', researcher:'fr-planner', 'security-auditor':'fr-security-reviewer', devops:'fr-shipper', 'cicd-engineer':'fr-shipper', 'api-docs':'fr-scriber' });
const BASE_VARIANTS = ['base', 'postask_native', 'postask_imit', 'postask_ruflo', 'sona_traj', 'all'];
const VARIANTS = [...BASE_VARIANTS, ...BASE_VARIANTS.filter(v => v !== 'postask_ruflo').map(v => v + '_minilm')];   // _minilm: CLAUDE_FLOW_ROUTER_EMBEDDER=minilm (ADR-390)
const argv = process.argv.slice(2); const arg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };

// exact one-sided binomial P(X >= k | n, p)
function lnC(n, k) { let s = 0; for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i); return s; }
export function binomUpper(k, n, p) { if (k <= 0) return 1; if (p <= 0) return 0; if (p >= 1) return 1; let s = 0; for (let x = k; x <= n; x++) s += Math.exp(lnC(n, x) + x * Math.log(p) + (n - x) * Math.log(1 - p)); return Math.min(1, s); }
function wilson(k, n, z = 1.959964) { if (!n) return [null, null]; const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [Math.max(0, c - h), Math.min(1, c + h)]; }

export function loadRows(file) {
  const raw = fs.readFileSync(file);
  const rows = raw.toString('utf8').trim().split('\n').map(JSON.parse).filter(r => ROLES.includes(r.actual_agent) && r.task_text).sort((a, b) => a.ts < b.ts ? -1 : 1);
  const cut = Math.floor(rows.length * 0.7);
  return { rows, train: rows.slice(0, cut), test: rows.slice(cut) };
}
const isPassed = r => r.outcome === 'passed';

async function runVariant(variantFull, dataFile) {
  const embedder = variantFull.endsWith('_minilm') ? 'minilm' : 'hash'; const variant = variantFull.replace(/_minilm$/, '');
  process.env.CLAUDE_FLOW_ROUTER_EMBEDDER = embedder;
  const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), `ruflo-${variantFull}-`));
  process.chdir(sandbox);                                   // BEFORE importing ruflo: it resolves .claude-flow/.swarm against cwd at import time
  const SRC = process.env.RUFLO_PKG_SRC;
  if (!SRC) throw new Error('set RUFLO_PKG_SRC to <...>/@claude-flow/cli/dist/src');
  // SONA learning in trajectory-end needs the memory DB ("Database not initialized. Run: claude-flow memory init"); `memory init` is the supported way, in the sandbox only
  const init = spawnSync(process.execPath, [path.join(SRC, '..', '..', 'bin', 'cli.js'), 'memory', 'init'], { cwd: sandbox, encoding: 'utf8', timeout: 120000 });
  const initOk = init.status === 0;
  const hooks = await import(pathToFileURL(path.join(SRC, 'mcp-tools', 'hooks-tools.js')).href);
  const { train, test, rows } = loadRows(dataFile);
  const errors = []; const note = (w, e) => { if (errors.length < 15) errors.push(`${w}: ${String(e).slice(0, 160)}`); };
  const routeOne = async (r) => {
    try {
      const out = await hooks.hooksRoute.handler({ task: r.task_text });
      const a = out?.primaryAgent?.type;
      return { agent: a ? (ROLES.includes(a) ? a : FROM_RUFLO[a] ?? `ruflo:${a}`) : null, raw: a, method: out?.routing?.method ?? null, pattern: out?.matchedPattern ?? null, conf: out?.primaryAgent?.confidence ?? null };
    } catch (e) { note('route', e); return { agent: null, raw: null, error: String(e) }; }
  };
  const routeAll = async (rs) => { const o = []; for (const r of rs) o.push(await routeOne(r)); return o; };
  const score = (rs, ps) => { const ok = rs.map((r, i) => [r, ps[i]]).filter(([, p]) => !p.error); const k = ok.filter(([r, p]) => p.agent === r.actual_agent).length; return { k, n: ok.length, acc: ok.length ? k / ok.length : null, errors: rs.length - ok.length }; };
  const before = await routeAll(test), trBefore = await routeAll(train);

  const useNative = ['postask_native', 'postask_imit', 'all'].includes(variant), useRuflo = variant === 'postask_ruflo', useTraj = ['sona_traj', 'all'].includes(variant);
  let n = 0, trained = 0, trajLearned = 0;
  for (const r of train) {
    n++;
    const ok = variant === 'postask_imit' ? true : isPassed(r);
    const q = ok ? 0.9 : (r.outcome === 'fix-round' || r.outcome === 'defects') ? 0.4 : 0.2;
    if (useNative || useRuflo) {
      try { const o = await hooks.hooksPostTask.handler({ taskId: `b${n}`, task: r.task_text, agent: useRuflo ? TO_RUFLO[r.actual_agent] : r.actual_agent, success: ok, quality: q }); trained++; if (o?.error) note('post-task', o.error); }
      catch (e) { note('post-task', e); }
    }
    if (useTraj) {
      try {
        const s = await hooks.hooksTrajectoryStart.handler({ task: r.task_text, agent: TO_RUFLO[r.actual_agent] });
        await hooks.hooksTrajectoryStep.handler({ trajectoryId: s.trajectoryId, action: `dispatch ${TO_RUFLO[r.actual_agent]}`, result: r.outcome, quality: q });
        const e = await hooks.hooksTrajectoryEnd.handler({ trajectoryId: s.trajectoryId, success: ok });
        if (e?.learning?.sonaUpdate) trajLearned++;
        if (n === 1) note('traj-end sample', JSON.stringify(e).slice(0, 150));
      } catch (e) { note('trajectory', e); }
    }
  }
  const after = await routeAll(test), trAfter = await routeAll(train);
  let sonaPatterns = null;
  try { const sp = JSON.parse(fs.readFileSync(path.join(sandbox, '.swarm', 'sona-patterns.json'), 'utf8')); const ps = Array.isArray(sp.patterns) ? sp.patterns : Object.values(sp.patterns ?? {}); const cs = ps.map(x => (Array.isArray(x) ? x[1] : x).confidence).filter(Number.isFinite); sonaPatterns = { n: cs.length, max_confidence: Math.max(0, ...cs), n_ge_0_6: cs.filter(c => c >= 0.6).length }; } catch { /* no SONA file for this variant */ }
  const sB = score(test, before), sA = score(test, after), tB = score(train, trBefore), tA = score(train, trAfter);
  const cnt = (rs) => rs.reduce((m, r) => { if (r.method) m[r.method] = (m[r.method] || 0) + 1; return m; }, {});
  const learnedHits = after.filter(p => (p.pattern || '').startsWith('learned-') || p.method === 'sona-pattern').length;
  const classes = ROLES.map(a => [a, test.filter(r => r.actual_agent === a).length]); const [maj, majN] = classes.sort((a, b) => b[1] - a[1])[0];
  const p0 = majN / test.length;
  const rule = test.filter(r => r.phase_rule === r.actual_agent).length;
  const [lo, hi] = wilson(sA.k, sA.n);
  process.chdir(HERE); try { fs.rmSync(sandbox, { recursive: true, force: true }); } catch { /* sandbox in %TEMP%: leaving it is harmless */ }
  return { variant: variantFull, embedder, sona_patterns: sonaPatterns, n_rows: rows.length, n_train: train.length, n_test: test.length, memory_init_ok: initOk, trained_rows: trained, sona_updates: trajLearned,
    test_acc_before: sB.acc, test_acc_after: sA.acc, test_k_after: sA.k, test_ci95_after: [lo, hi], train_acc_before: tB.acc, train_acc_after: tA.acc,
    majority_class: maj, majority_baseline_test: p0, phase_rule_test: rule / test.length, phase_rule_k: rule,
    changed_preds_test: test.filter((_, i) => before[i].agent !== after[i].agent).length, methods_after: cnt(after), learned_or_sona_hits_test: learnedHits,
    binomial_vs_majority: { k: sA.k, n: sA.n, p0, p_one_sided: sA.n ? binomUpper(sA.k, sA.n, p0) : null },
    errors, route_errors: sB.errors + sA.errors + tB.errors + tA.errors, sample_after: after.slice(0, 6).map(p => [p.raw, p.method, p.pattern]) };
}

async function main() {
  const dataFile = arg('--data', path.join(HERE, 'data', 'ruflo_dispatch_v2.jsonl'));
  if (argv.includes('--all')) {
    const outs = await Promise.all(VARIANTS.map(v => new Promise((res) => {
      const p = spawn(process.execPath, [fileURLToPath(import.meta.url), '--variant', v, '--data', dataFile], { env: process.env, stdio: ['ignore', 'pipe', 'inherit'] });
      let s = ''; p.stdout.on('data', d => s += d); p.on('close', () => { try { res(JSON.parse(s.trim().split('\n').filter(l => l.startsWith('{')).pop())); } catch { res({ variant: v, error: s.slice(-300) }); } });
    })));
    const ds = fs.readFileSync(dataFile);
    const res = { label: 'descriptive, not gated', dataset: path.relative(path.dirname(HERE), dataFile).split(path.sep).join('/'), variants: outs };
    fs.writeFileSync(path.join(HERE, 'ruflo_train2_result.json'), JSON.stringify(res, null, 1)); console.log(JSON.stringify(res, null, 1)); return;
  }
  const v = arg('--variant', 'base'); if (!VARIANTS.includes(v)) throw new Error(`variant must be one of ${VARIANTS}`);
  const r = await runVariant(v, dataFile); console.log(JSON.stringify(r));
  process.exit(0);
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) main().catch(e => { console.error(e); process.exit(1); });
