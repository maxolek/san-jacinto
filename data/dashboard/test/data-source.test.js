import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveDataSource } from '../src/data-source.js';

const page = 'https://example.org/san-jacinto/';
const digest = 'a'.repeat(64);
const manifest = { schema_version: 1, sha256: digest, snapshot: `snapshots/${digest}.duckdb`,
  size_bytes: 1000, exported_at: '2026-09-09T12:00:00Z', row_counts: {search_stats: 10} };
const response = value => async () => new Response(JSON.stringify(value), {headers:{'content-type':'application/json'}});

test('public data resolves relative to the app subpath and bypasses manifest cache', async () => {
  let request;
  const source = await resolveDataSource(page, page, {fetcher: async (...args) => {
    request = args;
    return response(manifest)();
  }});
  assert.equal(request[0], page + 'data/manifest.json');
  assert.equal(request[1].cache, 'no-store');
  assert.equal(source.url, page + 'data/' + manifest.snapshot);
  assert.equal(source.manifest.row_counts.search_stats, 10);
});

test('explicit database URLs and local mode do not fetch a manifest', async () => {
  const fetcher = () => { throw Error('unexpected fetch'); };
  assert.equal((await resolveDataSource(page+'?db=data/test.duckdb', page, {fetcher})).url, page+'data/test.duckdb');
  assert.equal((await resolveDataSource(page+'?db=https://data.example.org/db.duckdb', page, {fetcher})).url, 'https://data.example.org/db.duckdb');
  assert.equal(await resolveDataSource(page+'?local=1', page, {fetcher}), null);
  await assert.rejects(resolveDataSource(page+'?db=file:///tmp/db.duckdb', page, {fetcher}), /HTTP/);
});

test('missing publication is an error in production and a file picker in development', async () => {
  const fetcher = async () => new Response('', {status:404});
  await assert.rejects(resolveDataSource(page,page,{fetcher}), /unavailable/);
  assert.equal(await resolveDataSource(page,page,{fetcher,development:true}), null);
  const html = async () => new Response('<html></html>', {headers:{'content-type':'text/html'}});
  assert.equal(await resolveDataSource(page,page,{fetcher:html,development:true}), null);
  await assert.rejects(resolveDataSource(page,page,{fetcher:html}), /invalid/);
});

test('rejects malformed manifests and snapshot paths outside the publication', async () => {
  for (const value of [null, {}, {...manifest,snapshot:'https://unrelated.example/db.duckdb'},
    {...manifest,snapshot:'../db.duckdb'}, {...manifest,size_bytes:0}, {...manifest,exported_at:'bad'},
    {...manifest,row_counts:{search_stats:0}}]) {
    await assert.rejects(resolveDataSource(page,page,{fetcher:response(value)}), /invalid/);
  }
});
