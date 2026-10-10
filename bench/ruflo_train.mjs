// bench/ruflo_train.mjs - DESCRIPTIVE, NOT GATED: train ruflo's learning layer on real freight-recovery dispatch outcomes and
// measure `hooks route` before/after on held-out rows (chronological split: first 70% train, last 30% test).
//
// Dataset: ONLY the committed snapshot bench/data/ruflo_dispatch.jsonl (70 rows; 69 fr-* dispatches + 1 distiller row that is
// filtered out). It is the router-eval dataset built from freight-recovery's dispatch history (RD-LOG R117: "Real 69-row replay
// of freight-recovery's dispatch history (router-eval/dataset.jsonl)"), source sha256 e3221eab..., reduced to the fields used
// here (ts, run, task_text, actual_agent, outcome) and passed through ingest.scrub (scrub-v1). Its sha256 is logged in the result.
// CLI: $RUFLO_CLI (path to @claude-flow/cli's bin/cli.js, run with node) or else the pinned `npx -y @claude-flow/cli@3.56.3`.
// The CLI runs with cwd = bench/work/rf (gitignored) so its .swarm/ / .claude-flow/ state never lands in the repo.
// A CLI call that fails (spawn error, timeout/signal, non-zero exit, or `hooks route` output without an `Agent:` line) is an
// ERROR, recorded in the result (route_errors, train_errors, errors[]) and excluded from the accuracy denominators; it is never
// counted as a wrong route. `complete` is false when any call failed.
// Result: bench/ruflo_train_result.json with the majority-class baseline and an exact one-sided binomial test of the test
// accuracy (over the rows that routed) against it. n_test = 21, so this is a smoke-level description, not evidence for or
// against ruflo routing.
// Usage: node bench/ruflo_train.mjs [--dry-run]     (--dry-run: dataset, split and baseline only; no CLI calls)
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto'; import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DS = path.join(HERE, 'data', 'ruflo_dispatch.jsonl');
const CWD = path.join(HERE, 'work', 'rf');
const CLI_PKG = '@claude-flow/cli@3.56.3';
export const ROLES = ['fr-planner','fr-builder','fr-tester','fr-reviewer','fr-scriber','fr-shipper','fr-ci-diagnoser','fr-security-reviewer'];
const TO_RUFLO = { 'fr-planner':'planner','fr-builder':'coder','fr-tester':'tester','fr-reviewer':'reviewer','fr-scriber':'documenter','fr-shipper':'cicd-engineer','fr-ci-diagnoser':'devops','fr-security-reviewer':'security-auditor' };
export const FROM_RUFLO = { coder:'fr-builder','backend-dev':'fr-builder','frontend-dev':'fr-builder','performance-engineer':'fr-builder',tester:'fr-tester',reviewer:'fr-reviewer',architect:'fr-planner',planner:'fr-planner',researcher:'fr-planner','security-architect':'fr-security-reviewer','security-auditor':'fr-security-reviewer',devops:'fr-shipper','cicd-engineer':'fr-shipper','api-docs':'fr-scriber',documenter:'fr-scriber' };
const Q = { passed:1, 'fix-round':.4, defects:.4, 're-dispatch':.2, blocked:.1 };

// ---- exact binomial: P(X >= k | n, p)
function lnC(n, k) { let s = 0; for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i); return s; }
export function binomUpper(k, n, p) {
  if (k <= 0) return 1; if (p <= 0) return 0; if (p >= 1) return 1;
  let s = 0; for (let x = k; x <= n; x++) s += Math.exp(lnC(n, x) + x * Math.log(p) + (n - x) * Math.log(1 - p)); return Math.min(1, s);
}

// ---- the ruflo CLI, pinned, without a shell (task texts go through argv untouched)
function cliCommand() {
  if (process.env.RUFLO_CLI) return [process.execPath, [process.env.RUFLO_CLI]];
  const npxCli = path.join(path.dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npx-cli.js');
  if (fs.existsSync(npxCli)) return [process.execPath, [npxCli, '-y', CLI_PKG]];
  return ['npx', ['-y', CLI_PKG]];
}

// one CLI call -> {ok, out, error}. ok only for a clean exit 0; everything else is an error, never an answer.
export function runCli(args, spawn = spawnSync, cmd = cliCommand()) {
  const [exe, pre] = cmd;
  let p;
  try { p = spawn(exe, [...pre, ...args], { cwd: CWD, encoding: 'utf8', timeout: 180000 }); }
  catch (e) { return { ok: false, out: '', error: `spawn: ${e.message}` }; }
  const out = (p.stdout || '') + (p.stderr || '');
  if (p.error) return { ok: false, out, error: `spawn: ${p.error.code || p.error.message}` };
  if (p.signal) return { ok: false, out, error: `killed by ${p.signal}` };
  if (p.status !== 0) return { ok: false, out, error: `exit ${p.status}: ${out.trim().slice(-160)}` };
  return { ok: true, out, error: null };
}

// `hooks route` -> {agent, error}: agent is the routed ruflo agent, or null with an error when the call failed / did not route
export function route(task, cli = runCli) {
  const r = cli(['hooks', 'route', '--task', task]);
  if (!r.ok) return { agent: null, error: r.error };
  const agent = r.out.match(/\|\s*Agent:\s*([\w-]+)/)?.[1] ?? null;
  return agent ? { agent, error: null } : { agent: null, error: 'no Agent: line in route output' };
}

// accuracy over the rows that routed; errors are counted separately
export function score(rows, preds) {
  const ok = rows.map((r, i) => [r, preds[i]]).filter(([, p]) => !p.error);
  const k = ok.filter(([r, p]) => FROM_RUFLO[p.agent] === r.actual_agent).length;
  return { acc: ok.length ? k / ok.length : null, k, n: ok.length, errors: rows.length - ok.length };
}

export function loadDataset(file = DS) {
  const raw = fs.readFileSync(file);
  const rows = raw.toString('utf8').trim().split('\n').map(JSON.parse).filter(r => ROLES.includes(r.actual_agent) && r.task_text).sort((a,b)=>a.ts<b.ts?-1:1);
  const cut = Math.floor(rows.length*.7);
  return { sha: crypto.createHash('sha256').update(raw).digest('hex'), rows, train: rows.slice(0,cut), test: rows.slice(cut) };
}

export function evaluate({ train, test }, cli = runCli) {
  const errors = [];
  const note = (where, e) => { if (e && errors.length < 20) errors.push(`${where}: ${e}`); };
  const routeAll = (rs, where) => rs.map((r, i) => { const p = route(r.task_text, cli); note(`${where}[${i}]`, p.error); return p; });
  const before = routeAll(test, 'route test before'), trBefore = routeAll(train, 'route train before');
  let n = 0, trainErrors = 0;
  for (const r of train) {
    const ok = r.outcome==='passed'; const q = Q[r.outcome] ?? .6;
    for (const args of [['hooks','post-task','-i',`bench-${n++}`,'--success',String(ok),'-q',String(q),'-t',r.task_text,'-a',TO_RUFLO[r.actual_agent],'--store-results'],
                        ['hooks','model-outcome','--task',r.task_text,'--model','sonnet','--outcome',ok?'success':'failure']]) {
      const res = cli(args);
      if (!res.ok) { trainErrors++; note(args.slice(0, 2).join(' '), res.error); }
    }
  }
  const tr = cli(['hooks','intelligence','--train']);
  if (!tr.ok) { trainErrors++; note('hooks intelligence --train', tr.error); }
  const after = routeAll(test, 'route test after'), trAfter = routeAll(train, 'route train after');
  const sB = score(test, before), sA = score(test, after), tB = score(train, trBefore), tA = score(train, trAfter);
  const routeErrors = sB.errors + sA.errors + tB.errors + tA.errors;
  return { before, after, sB, sA, tB, tA, trainErrors, routeErrors, errors, trainOut: (tr.out || '').slice(0, 300),
           changed: test.filter((_, i) => !before[i].error && !after[i].error && before[i].agent !== after[i].agent).length };
}

export function majorityOf(test) {
  const majorityCount = Math.max(...ROLES.map(a=>test.filter(r=>r.actual_agent===a).length));
  return { p: majorityCount / test.length, cls: ROLES.find(a=>test.filter(r=>r.actual_agent===a).length===majorityCount) };
}

export function result(base, ev) {
  return { ...base, complete: ev.routeErrors === 0 && ev.trainErrors === 0,
    test_acc_before: ev.sB.acc, test_acc_after: ev.sA.acc, train_acc_before: ev.tB.acc, train_acc_after: ev.tA.acc,
    n_test_routed_before: ev.sB.n, n_test_routed_after: ev.sA.n, route_errors: ev.routeErrors, train_errors: ev.trainErrors, errors: ev.errors,
    changed_preds_test: ev.changed,
    binomial_test: { h1: 'test_acc_after > majority_baseline_test (rows that routed)', k: ev.sA.k, n: ev.sA.n, p0: base.majority_baseline_test,
                     p_one_sided: ev.sA.n ? binomUpper(ev.sA.k, ev.sA.n, base.majority_baseline_test) : null },
    train_cmd_tail: ev.trainOut };
}

function main() {
  const DRY = process.argv.includes('--dry-run');
  const ds = loadDataset();
  const m = majorityOf(ds.test);
  const base = { label: 'descriptive, not gated', dataset: path.relative(path.dirname(HERE), DS).split(path.sep).join('/'), dataset_sha256: ds.sha,
    cli: process.env.RUFLO_CLI ? `RUFLO_CLI=${process.env.RUFLO_CLI}` : `npx -y ${CLI_PKG}`, cwd: 'bench/work/rf', n_rows: ds.rows.length, n_train: ds.train.length,
    n_test: ds.test.length, majority_baseline_test: m.p, majority_class_test: m.cls };
  console.error(`dataset ${base.dataset} sha256=${ds.sha} rows=${ds.rows.length} train=${ds.train.length} test=${ds.test.length} majority=${m.p.toFixed(3)}`);
  if (DRY) { console.log(JSON.stringify(base, null, 1)); return 0; }
  fs.mkdirSync(CWD, { recursive: true });
  const res = result(base, evaluate(ds));
  fs.writeFileSync(path.join(HERE,'ruflo_train_result.json'), JSON.stringify(res,null,1)); console.log(JSON.stringify(res,null,1));
  if (!res.complete) console.error(`incomplete: ${res.route_errors} route errors, ${res.train_errors} train errors (see errors[]); not counted as wrong routes`);
  return res.complete ? 0 : 3;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) process.exitCode = main();
