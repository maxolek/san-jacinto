"""Log values as stored in a column of a given declared type (SQLite type affinity).

SQLite keeps any value in any column, but the analytics load reads the raw database
through DuckDB, which rejects a whole table when a column holds a value of another
type (2.5 in an INTEGER column, a list in a REAL one). So a value that doesn't fit
its column is never written there: callers keep it in the row's extra_fields JSON.
"""
import json
import math

MISMATCH = object()  # coerce(): the value doesn't fit the column


def affinity(declared_type):
    """SQLite's affinity of a declared column type: INTEGER, TEXT, BLOB, REAL or NUMERIC."""
    t = (declared_type or "").upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if not t or "BLOB" in t:
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


def coerce(value, declared_type):
    """`value` as stored in a column of `declared_type`: None stays None, a whole-number
    float becomes an int for INTEGER, lists/objects become JSON text for TEXT columns.
    Returns MISMATCH when the value doesn't fit (e.g. 2.5 or "x" for INTEGER, or an
    integer beyond SQLite's 64 bits)."""
    if value is None:
        return None
    kind = affinity(declared_type)
    if kind == "TEXT":
        return value if isinstance(value, str) else json.dumps(value)
    if kind == "BLOB":  # no declared type: stored as given
        return json.dumps(value) if isinstance(value, (list, dict)) else value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and kind != "REAL":
        value = int(value)
    if isinstance(value, int):
        if kind == "REAL":
            return float(value)
        return value if -2**63 <= value < 2**63 else MISMATCH
    if isinstance(value, float) and math.isfinite(value) and kind in ("REAL", "NUMERIC"):
        return value
    return MISMATCH
