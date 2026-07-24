#!/usr/bin/env python3
"""Fetch a slice of the public phh-dataset (uoftcprg/phh-dataset, MIT) for
bulk-volume bootstrapping. Downloads via the GitHub API (using the `gh`
CLI's existing auth) rather than being committed to the repo, since it's
large and trivially reproducible - see data/samples/README.md.

Usage: python scripts/fetch_phh_bulk.py [n_files]
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

STAKE_DIR = "data/handhq/PS-2009-07-01_2009-07-23_25NLH_OBFU/0.25"
REPO = "uoftcprg/phh-dataset"
DEST = Path(__file__).resolve().parent.parent / "data" / "samples" / "phh" / "bulk"


def gh_api(path: str) -> dict | list:
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main(n_files: int) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    listing = gh_api(f"repos/{REPO}/contents/{STAKE_DIR}")
    names = sorted(entry["path"].rsplit("/", 1)[-1] for entry in listing)[:n_files]

    for name in names:
        dest_path = DEST / name
        if dest_path.exists():
            print(f"skip (already have) {name}")
            continue
        entry = gh_api(f"repos/{REPO}/contents/{STAKE_DIR}/{name}")
        content = base64.b64decode(entry["content"])
        dest_path.write_bytes(content)
        print(f"fetched {name} ({len(content)} bytes)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)
