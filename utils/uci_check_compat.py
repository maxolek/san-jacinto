"""Launch a UCI engine while normalizing legacy boolean check-option defaults."""
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

CHECK_DEFAULT = re.compile(rb"^(option name .*? type check\b.*?\bdefault\s+)(0|1)(?=\s|$)")


def main():
    if len(sys.argv) < 2:
        print("usage: uci_check_compat.py ENGINE [ARGS...]", file=sys.stderr)
        return 2

    command = sys.argv[1:]
    executable = Path(command[0]).resolve()
    command[0] = str(executable)
    process = subprocess.Popen(
        command,
        cwd=executable.parent,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    def forward_input():
        try:
            for line in sys.stdin.buffer:
                process.stdin.write(line)
                process.stdin.flush()
        except BrokenPipeError:
            pass
        finally:
            if process.stdin:
                process.stdin.close()

    def terminate(_signal, _frame):
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, terminate)
    threading.Thread(target=forward_input, daemon=True).start()

    for line in process.stdout:
        line = CHECK_DEFAULT.sub(
            lambda match: match.group(1) + (b"false" if match.group(2) == b"0" else b"true"),
            line,
        )
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()

    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
