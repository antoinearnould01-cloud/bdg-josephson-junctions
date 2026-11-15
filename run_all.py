"""Regenerate every figure in figures/ (from the cached data in data/ when present).

    python run_all.py               # all figures, using cached data
    python run_all.py ch5 ch7       # only scripts whose name starts with ch5 or ch7
    python run_all.py --recompute   # rerun every simulation (many hours)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    args = sys.argv[1:]
    extra = [a for a in args if a.startswith("--")]
    prefixes = [a for a in args if not a.startswith("--")]
    scripts = sorted(p for p in (ROOT / "scripts").glob("ch*.py")
                     if not prefixes or any(p.name.startswith(x) for x in prefixes))
    for script in scripts:
        print(f"=== {script.name}", flush=True)
        subprocess.run([sys.executable, str(script), *extra], check=True)


if __name__ == "__main__":
    main()
