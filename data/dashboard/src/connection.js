/**
 * DuckDB-WASM connection manager for the chess analytics dashboard.
 * 
 * Provides helper query functions that work through the Mosaic coordinator.
 * Database initialization is handled in main.js.
 */
import { coordinator } from '@uwdata/mosaic-core';

/**
 * Run a raw SQL query and return results as an array of objects.
 */
export async function query(sql) {
  const result = await coordinator().query(sql);
  return result;
}

/**
 * Get table names in the database.
 */
export async function getTables() {
  const result = await coordinator().query(`SHOW TABLES`);
  const arr = Array.from(result);
  return arr.map(r => r.name);
}

/**
 * Get row count for a table.
 */
export async function getCount(table) {
  const result = await coordinator().query(`SELECT COUNT(*) as n FROM ${table}`);
  const arr = Array.from(result);
  return arr[0]?.n ?? 0;
}

/**
 * Check if a table exists.
 */
export async function tableExists(table) {
  try {
    const tables = await getTables();
    return tables.includes(table);
  } catch {
    return false;
  }
}

export function isReady() { return _ready; }
