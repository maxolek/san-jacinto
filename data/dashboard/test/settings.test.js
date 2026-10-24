import test from 'node:test';
import assert from 'node:assert/strict';
import { coreFilter, coreRows, coreType, onCoreType, onSettings, openingLabelSQL, openingMode, setCoreType, setCoreTypes,
  setOpeningMode } from '../src/settings.js';

test('core type only takes values the snapshot offers, and drives the SQL predicates', () => {
  assert.equal(coreType(), 'all');
  assert.equal(coreRows(), "core_type = 'all'");
  assert.equal(coreFilter(), 'TRUE');
  const seen = [];
  const off = onCoreType(s => seen.push(s.coreType));
  setCoreType('P');
  assert.equal(coreType(), 'all', 'P is not offered before setCoreTypes');
  setCoreTypes(['P', 'E', '-']);
  setCoreType('P');
  assert.equal(coreRows(), "core_type = 'P'");
  assert.equal(coreFilter('g.core_type'), "g.core_type = 'P'");
  setCoreType("P'; DROP TABLE game_stats; --");
  assert.equal(coreType(), 'all', 'anything unknown falls back to all');
  setCoreType('-');
  assert.equal(coreFilter(), "core_type = '-'");
  setCoreTypes(['P']);
  assert.equal(coreType(), 'all', 'a core type the new snapshot lacks is dropped');
  off();
  setCoreType('P');
  assert.deepEqual(seen, ['P', 'all', '-']);
  setCoreType('all');
});

test('opening labels follow the family / variation toggle and notify every-setting listeners', () => {
  const keys = [];
  const off = onSettings(() => keys.push(openingMode()));
  assert.equal(openingLabelSQL('o'), 'o.opening');
  setOpeningMode('variation');
  assert.match(openingLabelSQL(), /opening_variation = '' THEN opening \|\| ' \(main line\)'/);
  setOpeningMode('nonsense');
  assert.equal(openingMode(), 'variation');
  setOpeningMode('family');
  off();
  assert.deepEqual(keys, ['variation', 'family']);
});
