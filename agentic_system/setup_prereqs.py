"""Install project prerequisites before running the app.

This script is intended to be run once on a developer machine before launching the
Agentic Workday OS project. It installs the Python packages declared in requirements.txt
and prints a short status summary.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQ_FILE = ROOT / "requirements.txt"


def run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    print("Agentic Workday OS prerequisite installer")
    print(f"Project root: {ROOT}")

    python_exe = sys.executable
    if not REQ_FILE.exists():
        raise SystemExit(f"Requirements file not found: {REQ_FILE}")

    print("Installing Python packages from requirements.txt...")
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    run([python_exe, "-m", "pip", "install", "-r", str(REQ_FILE)])
    print("Installing the Chromium browser used for NTULearn SSO...")
    run([python_exe, "-m", "playwright", "install", "chromium"])

    print("\nPrerequisites installed successfully.")
    print("Next steps:")
    print("  1. Copy .env.example to .env and fill in your real credentials")
    print("  2. Set ENABLE_OUTLOOK=true, ENABLE_AWS=true, etc. as needed")
    print("  3. Run: .\\run_app.ps1")


if __name__ == "__main__":
    main()
