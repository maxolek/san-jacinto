"""Small helper utilities."""
import json
import shutil
from pathlib import Path

def safe_val(val):
    """Convert lists/dicts to JSON strings for DB insertion."""
    if isinstance(val, list) or isinstance(val, dict):
        return json.dumps(val)
    return val


def safe(arr, i):
    """Safe index into array, returning 0 if out of bounds."""
    return arr[i] if i < len(arr) else 0

# convert search_{pid}.jsonl to search.jsonl, game_{pid}.jsonl to game.jsonl, etc.
# search_{pid} is the process specific file, search is the expected readable file
def consolidate_instance_logs(logroot):
    logroot = Path(logroot)
    for basename in ("game", "search", "timing", "root_moves"):
        parts = sorted(logroot.glob(f"{basename}_*.jsonl"))
        if not parts:
            continue

        output_path = logroot / f"{basename}.jsonl"
        with open(output_path, "w", encoding="utf-8") as out_f:
            for part in parts:
                with open(part, "r", encoding="utf-8") as in_f:
                    shutil.copyfileobj(in_f, out_f)