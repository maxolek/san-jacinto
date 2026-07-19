"""
Entry point: python -m data.dashboard
Starts the Vite dev server for the analytics dashboard.
"""
import subprocess
import sys
import webbrowser
import threading
from pathlib import Path


def main():
    dashboard_dir = Path(__file__).parent
    # Ensure node_modules exist
    if not (dashboard_dir / "node_modules").exists():
        print("Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=dashboard_dir, check=True, shell=True)

    # Open browser after a short delay
    threading.Timer(2.0, webbrowser.open, args=["http://localhost:5173"]).start()

    # Start vite dev server (blocks until ctrl-c)
    try:
        subprocess.run(["npm", "run", "dev"], cwd=dashboard_dir, check=True, shell=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
