"""
Entry point: python -m data.dashboard
Starts the Vite dev server for the analytics dashboard (it opens the browser itself).
"""
import shutil
import subprocess
from pathlib import Path


def main():
    dashboard_dir = Path(__file__).parent
    # npm is npm.cmd on Windows; resolving it avoids shell=True on every platform
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm was not found; install Node.js 18 or newer")
    if not (dashboard_dir / "node_modules").exists():
        print("Installing dependencies...")
        subprocess.run([npm, "ci"], cwd=dashboard_dir, check=True)
    try:
        subprocess.run([npm, "run", "dev"], cwd=dashboard_dir, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
