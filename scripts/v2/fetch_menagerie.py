"""Clone MuJoCo Menagerie and check out the commit pinned in configs/v2/menagerie.lock.

Usage: uv run python scripts/v2/fetch_menagerie.py [--path DIR] [--update]
--path defaults to $MENAGERIE_PATH, else the lock's path. --update pins the checkout's current HEAD instead.
"""
import argparse
import os
import subprocess
from pathlib import Path

import yaml

LOCK = Path(__file__).resolve().parent.parent.parent / "configs" / "v2" / "menagerie.lock"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    lock = yaml.safe_load(LOCK.read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.environ.get("MENAGERIE_PATH", lock["path"]))
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    path = Path(a.path)
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git("clone", lock["repo"], str(path))
    if a.update:
        head = git("-C", str(path), "rev-parse", "HEAD")
        LOCK.write_text(LOCK.read_text().replace(lock["commit"], head))
        print(f"pinned {head}")
        return
    if git("-C", str(path), "rev-parse", "HEAD") != lock["commit"]:
        git("-C", str(path), "fetch", "origin")
        git("-C", str(path), "checkout", "--detach", lock["commit"])
    print(f"{path} at {git('-C', str(path), 'rev-parse', 'HEAD')}")


if __name__ == "__main__":
    main()
