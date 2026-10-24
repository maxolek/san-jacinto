import test from 'node:test';
import assert from 'node:assert/strict';
import { coreNote, coreOptions, DEFAULT_ENGINE_LIMIT, defaultBaseline, defaultSelection, formatHash, parseHash, pinnedCores,
  sanitizeBaseline, sanitizeSelection } from '../src/state.js';

test('view state round-trips through the URL hash', () => {
  const state = { tab: 'compare', engines: [1, 3, 5], base: 3, cores: 'P', open: 'variation' };
  assert.equal(formatHash(state), '#tab=compare&engines=1,3,5&base=3&cores=P&open=variation');
  assert.deepEqual(parseHash(formatHash(state)), state);
  assert.equal(formatHash({ tab: 'overview', cores: 'all', open: 'family' }), '#tab=overview');
  assert.deepEqual(parseHash(''), { tab: null, engines: null, base: null, cores: null, open: null });
  assert.deepEqual(parseHash('#engines=2,x,-1,4&open=weird').engines, [2, 4]);
  assert.equal(parseHash('#open=weird').open, null);
  assert.equal(parseHash('#cores=unknown').cores, null);
  assert.equal(parseHash(formatHash({ cores: '-' })).cores, '-');
});

test('selection defaults and sanitising', () => {
  assert.equal(DEFAULT_ENGINE_LIMIT, 8);
  assert.deepEqual(defaultSelection([3, 1, 2]), [1, 2, 3]);
  const many = Array.from({ length: 12 }, (_, i) => i + 1);
  assert.deepEqual(defaultSelection(many), [5, 6, 7, 8, 9, 10, 11, 12]);
  assert.deepEqual(sanitizeSelection([5, 2, 2, 99], [1, 2, 3, 4, 5]), [2, 5]);
  assert.deepEqual(sanitizeSelection([99], [1, 2]), [1, 2]);
  assert.deepEqual(sanitizeSelection(null, [1, 2]), [1, 2]);
});

test('baseline is the previous version of the newest selected engine unless chosen', () => {
  assert.equal(defaultBaseline([4, 2, 9]), 4);
  assert.equal(defaultBaseline([7]), 7);
  assert.equal(defaultBaseline([]), null);
  assert.equal(sanitizeBaseline(2, [2, 4, 9]), 2);
  assert.equal(sanitizeBaseline(3, [2, 4, 9]), 4);
});

test('core-type filter offers pinned core types only, and explains when there is no choice', () => {
  const values = options => options.map(o => o.value);
  assert.deepEqual(pinnedCores(['all', 'E', 'unknown', 'P', '-', 'mixed']), ['P', 'E', '-']);
  assert.deepEqual(values(coreOptions(['all', 'E', 'unknown', 'P', '-', 'mixed'])), ['all', 'P', 'E', '-']);
  assert.deepEqual(coreOptions(['all', 'P', 'E']).map(o => o.label), ['All', 'P-cores', 'E-cores']);
  assert.deepEqual(values(coreOptions(['all', 'P', 'unknown'])), ['all', 'P']);
  assert.equal(coreNote(['all', 'P', 'E']), null);
  assert.deepEqual(values(coreOptions(['all', 'P'])), ['all']);
  assert.equal(coreNote(['all', 'P']), 'every game ran on P-cores');
  assert.equal(coreNote(['all', 'unknown']), 'not recorded in this snapshot');
  assert.equal(coreNote([]), 'not recorded in this snapshot');
});
