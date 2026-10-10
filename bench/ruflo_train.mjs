// bench/ruflo_train.mjs - train ruflo's learning layer on REAL freight-recovery dispatch outcomes and
// measure hooks route before/after on held-out rows (chronological split: first 70% train, last 30% test).
// Usage: node bench/ruflo_train.mjs <dataset.jsonl> [--epochs N] ; ruflo state is isolated in bench/work/rf (cwd).
import fs from 'node:fs'; import path from 'node:path'; import { spawnSync } from 'node:child_process';
const CLI = process.env.RUFLO_CLI || 'C:/Users/mariu/AppData/Local/npm-cache/_npx/85fb20e3e7e3a233/node_modules/@claude-flow/cli/bin/cli.js';
const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, '$1'));
const CWD = path.join(HERE, 'work', 'rf'); fs.mkdirSync(CWD, { recursive: true });
const ds = process.argv[2];
const ROLES = ['fr-planner','fr-builder','fr-tester','fr-reviewer','fr-scriber','fr-shipper','fr-ci-diagnoser','fr-security-reviewer'];
const TO_RUFLO = { 'fr-planner':'planner','fr-builder':'coder','fr-tester':'tester','fr-reviewer':'reviewer','fr-scriber':'documenter','fr-shipper':'cicd-engineer','fr-ci-diagnoser':'devops','fr-security-reviewer':'security-auditor' };
const FROM_RUFLO = { coder:'fr-builder','backend-dev':'fr-builder','frontend-dev':'fr-builder','performance-engineer':'fr-builder',tester:'fr-tester',reviewer:'fr-reviewer',architect:'fr-planner',planner:'fr-planner',researcher:'fr-planner','security-architect':'fr-security-reviewer','security-auditor':'fr-security-reviewer',devops:'fr-shipper','cicd-engineer':'fr-shipper','api-docs':'fr-scriber',documenter:'fr-scriber' };
const Q = { passed:1, 'fix-round':.4, defects:.4, 're-dispatch':.2, blocked:.1 };
const run = (a) => { const p = spawnSync(process.execPath, [CLI, ...a], { cwd: CWD, encoding: 'utf8', timeout: 180000 }); return (p.stdout||'')+(p.stderr||''); };
const route = (t) => run(['hooks','route','--task',t]).match(/\|\s*Agent:\s*([\w-]+)/)?.[1] ?? null;
const rows = fs.readFileSync(ds,'utf8').trim().split('\n').map(JSON.parse).filter(r => ROLES.includes(r.actual_agent) && r.task_text).sort((a,b)=>a.ts<b.ts?-1:1);
const cut = Math.floor(rows.length*.7), train = rows.slice(0,cut), test = rows.slice(cut);
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
const res = { n_train:train.length, n_test:test.length, test_acc_before:acc(test,before), test_acc_after:acc(test,after), train_acc_before:acc(train,trBefore), train_acc_after:acc(train,trAfter),
  changed_preds_test: test.filter((_,i)=>before[i]!==after[i]).length, majority_baseline_test: Math.max(...ROLES.map(a=>test.filter(r=>r.actual_agent===a).length))/test.length, train_cmd_tail: trainOut };
fs.writeFileSync(path.join(HERE,'ruflo_train_result.json'), JSON.stringify(res,null,1)); console.log(JSON.stringify(res,null,1));
