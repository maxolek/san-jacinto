"""Bring the raw database up to the current schema on first use.

The schema is the base tables of init_raw_db.py plus every migration in
data/databases/migrate_schema.py (provenance, rating and net tables, new search
statistics, ...). Each migration is applied once and recorded in the database's
schema_migrations table. init_raw_db calls ensure_schema after creating the base
tables, and the programs that add rows call it first, so an existing database is
upgraded in place: its rows are kept, and columns added later are NULL for them.
"""
from ..databases import migrate_schema


def ensure_schema(cnxn, recheck=False):
    """Apply the pending raw migrations through this sqlite3 connection.

    Once the database is current this is one query. When something is applied, a
    transaction the caller left open is committed first, so call it before writing.
    recheck (init_raw_db): also re-apply what's missing of applied migrations.
    """
    if not recheck and not migrate_schema.pending_ids(cnxn):
        return
    results = migrate_schema.migrate_connection(cnxn, recheck=recheck)
    applied = [mid for mid, status in results.items() if status == "applied"]
    if applied:
        print(f"[DB] schema upgraded: {', '.join(applied)}")
