import test from 'node:test';
import assert from 'node:assert/strict';
import { assignColors, divergingColor, engineLabel, interpolate, OKABE_ITO, SEQUENTIAL, sequentialColor, textOn } from '../src/palette.js';

test('engine labels read "name version" and fall back to the id', () => {
  assert.equal(engineLabel({ id: 3, name: 'Tomahawk', version: '1.10.0' }), 'Tomahawk 1.10.0');
  assert.equal(engineLabel({ id: 3, name: 'Tomahawk', version: null }), 'Tomahawk');
  assert.equal(engineLabel({ id: 7 }), 'Engine 7');
});

test('colours follow registration order, not alphabetical version order', () => {
  const engines = [
    { id: 3, name: 'Tomahawk', version: '1.10.0' },
    { id: 1, name: 'Tomahawk', version: '1.8.0' },
    { id: 2, name: 'Tomahawk', version: '1.9.0' },
  ];
  const out = assignColors(engines);
  assert.deepEqual(out.map(e => e.label), ['Tomahawk 1.8.0', 'Tomahawk 1.9.0', 'Tomahawk 1.10.0']);
  assert.deepEqual(out.map(e => e.color), OKABE_ITO.slice(0, 3));
  assert.deepEqual(out.map(e => e.order), [0, 1, 2]);
  // the same engine keeps its colour whatever else is in the list order
  assert.equal(assignColors([...engines].reverse()).find(e => e.id === 3).color, OKABE_ITO[2]);
});

test('palette cycles, keeps labels unique and lightens black in dark mode', () => {
  const many = Array.from({ length: 10 }, (_, i) => ({ id: i + 1, name: 'X', version: String(i + 1) }));
  const out = assignColors(many);
  assert.equal(out[8].color, OKABE_ITO[0]);
  assert.equal(new Set(OKABE_ITO).size, 8);
  const dark = assignColors(many, { dark: true });
  assert.notEqual(dark[7].color, '#000000');
  const dupes = assignColors([{ id: 1, name: 'A', version: '1' }, { id: 2, name: 'A', version: '1' }]);
  assert.notEqual(dupes[0].label, dupes[1].label);
});

test('sequential and diverging scales interpolate between their stops', () => {
  assert.equal(sequentialColor(0), SEQUENTIAL[0].toLowerCase());
  assert.equal(sequentialColor(1), SEQUENTIAL[SEQUENTIAL.length - 1].toLowerCase());
  assert.equal(interpolate(['#000000', '#ffffff'], 0.5), '#808080');
  assert.equal(divergingColor(0), '#f3eee4');
  assert.equal(divergingColor(-5), divergingColor(-1));
  assert.equal(textOn('#ffffff'), '#2B2620');
  assert.equal(textOn('#2f5d7c'), '#FBF9F4');
});
