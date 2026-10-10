// bench/tests/ruflo_train.test.mjs - a ruflo CLI failure is an error, never a wrong route. Run: node --test bench/tests/
// (also run by test_ruflo_train.py under pytest). Stubbed CLI only: no ruflo, no network.
import test from 'node:test'; import assert from 'node:assert/strict';
import { runCli, route, score, evaluate, result, majorityOf, loadDataset, binomUpper } from '../ruflo_train.mjs';

const CMD = ['node', []];
const spawnOf = (p) => () => ({ stdout: '', stderr: '', status: 0, signal: null, ...p });

test('runCli: only a clean exit 0 is ok', () => {
  assert.equal(runCli(['x'], spawnOf({ stdout: '| Agent: coder |' }), CMD).ok, true);
  assert.match(runCli(['x'], spawnOf({ status: 1, stderr: 'npm ERR! 404' }), CMD).error, /^exit 1: npm ERR! 404/);
  assert.match(runCli(['x'], spawnOf({ status: null, signal: 'SIGTERM' }), CMD).error, /killed by SIGTERM/);
  assert.match(runCli(['x'], spawnOf({ status: null, error: Object.assign(new Error('x'), { code: 'ENOENT' }) }), CMD).error, /spawn: ENOENT/);
  assert.match(runCli(['x'], () => { throw new Error('boom'); }, CMD).error, /spawn: boom/);
});

test('route: a failed call or an unparsable output is an error, not an agent', () => {
  assert.deepEqual(route('t', () => ({ ok: true, out: '| Agent: tester |' })), { agent: 'tester', error: null });
  assert.equal(route('t', () => ({ ok: false, out: '', error: 'exit 1' })).error, 'exit 1');
  assert.equal(route('t', () => ({ ok: true, out: 'Usage: ...' })).error, 'no Agent: line in route output');
});

test('score: errors are excluded from the denominator, not counted wrong', () => {
  const rows = [{ actual_agent: 'fr-builder' }, { actual_agent: 'fr-tester' }, { actual_agent: 'fr-tester' }];
  const preds = [{ agent: 'coder', error: null }, { agent: null, error: 'exit 1' }, { agent: 'coder', error: null }];
  assert.deepEqual(score(rows, preds), { acc: 0.5, k: 1, n: 2, errors: 1 });
  assert.deepEqual(score(rows, preds.map(() => ({ agent: null, error: 'x' }))), { acc: null, k: 0, n: 0, errors: 3 });
});

test('evaluate + result: a CLI that fails half the time is reported incomplete, accuracy only over routed rows', () => {
  const ds = loadDataset();
  let i = 0;
  const flaky = (args) => (args[1] === 'route' && (i++ % 2) ? { ok: false, out: '', error: 'exit 7' } : { ok: true, out: '| Agent: coder |', error: null });
  const ev = evaluate(ds, flaky);
  const m = majorityOf(ds.test);
  const res = result({ majority_baseline_test: m.p }, ev);
  assert.equal(res.complete, false);
  assert.ok(res.route_errors > 0 && res.train_errors === 0 && res.errors.length > 0);
  assert.equal(res.n_test_routed_after + ev.sA.errors, ds.test.length);
  const builders = ds.test.filter((r, j) => !ev.after[j].error && r.actual_agent === 'fr-builder').length;
  assert.equal(res.binomial_test.k, builders);                           // "coder" -> fr-builder; failed rows are not wrong rows
  assert.equal(res.test_acc_after, builders / res.n_test_routed_after);
  const clean = result({ majority_baseline_test: m.p }, evaluate(ds, () => ({ ok: true, out: '| Agent: coder |', error: null })));
  assert.equal(clean.complete, true); assert.equal(clean.route_errors, 0); assert.equal(clean.n_test_routed_after, ds.test.length);
  const down = result({ majority_baseline_test: m.p }, evaluate(ds, () => ({ ok: false, out: '', error: 'exit 1' })));
  assert.equal(down.test_acc_after, null); assert.equal(down.binomial_test.p_one_sided, null); assert.ok(down.train_errors > 0);
});

test('binomUpper', () => {
  assert.equal(binomUpper(0, 21, 0.4), 1);
  assert.ok(Math.abs(binomUpper(3, 21, 9 / 21) - 0.9989379806362747) < 1e-12);   // the committed result's p value
});
