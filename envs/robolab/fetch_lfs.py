"""Fetch git-LFS objects of a RoboLab checkout over HTTPS (no git-lfs binary needed).

usage: python3 fetch_lfs.py <robolab_checkout> <commit>
Replaces every LFS pointer file with its object from media.githubusercontent.com, checked by sha256 and size.
"""
import hashlib
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MEDIA = "https://media.githubusercontent.com/media/NVlabs/RoboLab"


def pointer(path):
    with open(path, "rb") as f:
        head = f.read(200)
    if not head.startswith(b"version https://git-lfs"):
        return None
    kv = dict(line.split(" ", 1) for line in head.decode().splitlines() if " " in line)
    return kv["oid"].split(":", 1)[1], int(kv["size"])


def fetch(root, commit, rel):
    p = pointer(os.path.join(root, rel))
    if p is None:
        return "skip"
    oid, size = p
    url = f"{MEDIA}/{commit}/{urllib.parse.quote(rel)}"
    tmp = os.path.join(root, rel + ".lfs_tmp")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != oid or os.path.getsize(tmp) != size:
        os.remove(tmp)
        return f"bad {rel}"
    os.replace(tmp, os.path.join(root, rel))
    return "ok"


def main(root, commit):
    rels = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    rels = [r for r in rels if r and os.path.isfile(os.path.join(root, r)) and pointer(os.path.join(root, r))]
    print(f"{len(rels)} lfs pointers", flush=True)
    bad = 0
    with ThreadPoolExecutor(16) as ex:
        for i, out in enumerate(ex.map(lambda r: fetch(root, commit, r), rels)):
            if out.startswith("bad"):
                bad += 1
                print(out, flush=True)
            if i % 100 == 0:
                print(f"{i}/{len(rels)}", flush=True)
    print(f"done, {bad} bad", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
