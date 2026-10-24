/**
 * Query helpers and the attached snapshot's schema. The snapshot is attached read-only as `db`;
 * everything the dashboard computes lives in temporary tables of the in-memory database.
 */
import { coordinator } from '@uwdata/mosaic-core';

/** Map of relation name -> Set of column names in the attached snapshot. */
export const schema = new Map();

export async function query(sql) {
  const rows = await coordinator().query(sql, { type: 'json' });
  return Array.from(rows, normalizeRow);
}

export async function exec(sql, options) {
  await coordinator().exec(sql, options);
}

// Arrow decoding can still yield BigInt (e.g. HUGEINT sums); charts and tables want plain numbers.
function normalizeRow(row) {
  for (const key in row) {
    const v = row[key];
    if (typeof v === 'bigint') row[key] = Number(v);
  }
  return row;
}

export const quoteIdent = name => '"' + String(name).replaceAll('"', '""') + '"';
export const quoteString = value => "'" + String(value).replaceAll("'", "''") + "'";
/** Fully qualified name of a relation in the attached snapshot. */
export const src = name => `db.main.${quoteIdent(name)}`;

export function has(table, ...columns) {
  const cols = schema.get(table);
  return Boolean(cols) && columns.every(c => cols.has(c));
}

/** First existing column among candidates, or null. */
export function pick(table, ...candidates) {
  const cols = schema.get(table);
  if (!cols) return null;
  return candidates.find(c => cols.has(c)) ?? null;
}

export async function loadSchema() {
  const rows = await query(`
    SELECT table_name, column_name FROM duckdb_columns()
    WHERE database_name = 'db' AND schema_name = 'main'`);
  schema.clear();
  for (const { table_name, column_name } of rows) {
    if (!schema.has(table_name)) schema.set(table_name, new Set());
    schema.get(table_name).add(column_name);
  }
  return schema;
}

export async function attach(url) {
  await exec(`ATTACH ${quoteString(url)} AS db (READ_ONLY)`);
  await loadSchema();
}
