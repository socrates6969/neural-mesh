// bench/ruflo_train.mjs - DESCRIPTIVE, NOT GATED: train ruflo's learning layer on real freight-recovery dispatch outcomes and
// measure `hooks route` before/after on held-out rows (chronological split: first 70% train, last 30% test).
//
// Dataset: ONLY the committed snapshot bench/data/ruflo_dispatch.jsonl (70 rows; 69 fr-* dispatches + 1 distiller row that is
// filtered out). It is the router-eval dataset built from freight-recovery's dispatch history (RD-LOG R117: "Real 69-row replay
// of freight-recovery's dispatch history (router-eval/dataset.jsonl)"), source sha256 e3221eab..., reduced to the fields used
// here (ts, run, task_text, actual_agent, outcome) and passed through ingest.scrub (scrub-v1). Its sha256 is logged in the result.
// CLI: $RUFLO_CLI (path to @claude-flow/cli's bin/cli.js, run with node) or else the pinned `npx -y @claude-flow/cli@3.56.3`.
// The CLI runs with cwd = bench/work/rf (gitignored) so its .swarm/ / .claude-flow/ state never lands in the repo.
// Result: bench/ruflo_train_result.json with the majority-class baseline and an exact one-sided binomial test of the test
// accuracy against it. n_test = 21, so this is a smoke-level description, not evidence for or against ruflo routing.
// Usage: node bench/ruflo_train.mjs [--dry-run]     (--dry-run: dataset, split and baseline only; no CLI calls)
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto'; import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DS = path.join(HERE, 'data', 'ruflo_dispatch.jsonl');
const CWD = path.join(HERE, 'work', 'rf');
const CLI_PKG = '@claude-flow/cli@3.56.3';
const DRY = process.argv.includes('--dry-run');
const ROLES = ['fr-planner','fr-builder','fr-tester','fr-reviewer','fr-scriber','fr-shipper','fr-ci-diagnoser','fr-security-reviewer'];
const TO_RUFLO = { 'fr-planner':'planner','fr-builder':'coder','fr-tester':'tester','fr-reviewer':'reviewer','fr-scriber':'documenter','fr-shipper':'cicd-engineer','fr-ci-diagnoser':'devops','fr-security-reviewer':'security-auditor' };
const FROM_RUFLO = { coder:'fr-builder','backend-dev':'fr-builder','frontend-dev':'fr-builder','performance-engineer':'fr-builder',tester:'fr-tester',reviewer:'fr-reviewer',architect:'fr-planner',planner:'fr-planner',researcher:'fr-planner','security-architect':'fr-security-reviewer','security-auditor':'fr-security-reviewer',devops:'fr-shipper','cicd-engineer':'fr-shipper','api-docs':'fr-scriber',documenter:'fr-scriber' };
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
const [EXE, PRE] = cliCommand();
const run = (a) => { const p = spawnSync(EXE, [...PRE, ...a], { cwd: CWD, encoding: 'utf8', timeout: 180000 }); return (p.stdout||'')+(p.stderr||''); };
const route = (t) => run(['hooks','route','--task',t]).match(/\|\s*Agent:\s*([\w-]+)/)?.[1] ?? null;

const raw = fs.readFileSync(DS);
const dsSha = crypto.createHash('sha256').update(raw).digest('hex');
const rows = raw.toString('utf8').trim().split('\n').map(JSON.parse).filter(r => ROLES.includes(r.actual_agent) && r.task_text).sort((a,b)=>a.ts<b.ts?-1:1);
const cut = Math.floor(rows.length*.7), train = rows.slice(0,cut), test = rows.slice(cut);
const majorityCount = Math.max(...ROLES.map(a=>test.filter(r=>r.actual_agent===a).length));
const majority = majorityCount / test.length;
const base = { label: 'descriptive, not gated', dataset: path.relative(path.dirname(HERE), DS).split(path.sep).join('/'), dataset_sha256: dsSha,
  cli: process.env.RUFLO_CLI ? `RUFLO_CLI=${process.env.RUFLO_CLI}` : `npx -y ${CLI_PKG}`, cwd: 'bench/work/rf', n_rows: rows.length, n_train: train.length, n_test: test.length,
  majority_baseline_test: majority, majority_class_test: ROLES.find(a=>test.filter(r=>r.actual_agent===a).length===majorityCount) };
console.error(`dataset ${base.dataset} sha256=${dsSha} rows=${rows.length} train=${train.length} test=${test.length} majority=${majority.toFixed(3)}`);
if (DRY) { console.log(JSON.stringify(base, null, 1)); process.exit(0); }

fs.mkdirSync(CWD, { recursive: true });
const acc = (rs, preds) => rs.filter((r,i)=>FROM_RUFLO[preds[i]]===r.actual_agent).length/rs.length;
const before = test.map(r=>route(r.task_text)); const trBefore = train.map(r=>route(r.task_text));
let n=0;
for (const r of train) {
  const ok = r.outcome==='passed'; const q = Q[r.outcome] ?? .6;
  run(['hooks','post-task','-i',`bench-${n++}`,'--success',String(ok),'-q',String(q),'-t',r.task_text,'-a',TO_RUFLO[r.actual_agent],'--store-results']);
  run(['hooks','model-outcome','--task',r.task_text,'--model','sonnet','--outcome',ok?'success':'failure']);
}
const trainOut = run(['hooks','intelligence','--train']).slice(0,300);
const after = test.map(r=>route(r.task_text)); const trAfter = train.map(r=>route(r.task_text));
const kAfter = test.filter((r,i)=>FROM_RUFLO[after[i]]===r.actual_agent).length;
const res = { ...base, test_acc_before:acc(test,before), test_acc_after:acc(test,after), train_acc_before:acc(train,trBefore), train_acc_after:acc(train,trAfter),
  changed_preds_test: test.filter((_,i)=>before[i]!==after[i]).length,
  binomial_test: { h1: 'test_acc_after > majority_baseline_test', k: kAfter, n: test.length, p0: majority, p_one_sided: binomUpper(kAfter, test.length, majority) },
  train_cmd_tail: trainOut };
fs.writeFileSync(path.join(HERE,'ruflo_train_result.json'), JSON.stringify(res,null,1)); console.log(JSON.stringify(res,null,1));
