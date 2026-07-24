#!/usr/bin/env python3
"""Parse every vendored/fetched bootstrap hand (raw-text fixtures + any
fetched phh bulk volume) into data/poker.db. Re-runnable: duplicate
(site, hand_id) pairs across overlapping fetches are just skipped.
"""
from pathlib import Path

from poker_analytics import db
from poker_analytics.parsers.phh import parse_phhs_file
from poker_analytics.parsers.pokerstars_text import parse_file

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "poker.db"


def main() -> None:
    conn = db.connect(DB_PATH)
    inserted, skipped = 0, 0

    for f in sorted((ROOT / "data" / "samples" / "raw_text").glob("*.txt")):
        for hand in parse_file(f):
            try:
                db.insert_hand(conn, hand)
                inserted += 1
            except Exception:
                skipped += 1

    bulk_dir = ROOT / "data" / "samples" / "phh" / "bulk"
    if bulk_dir.exists() and any(bulk_dir.glob("*.phhs")):
        for f in sorted(bulk_dir.glob("*.phhs")):
            for hand in parse_phhs_file(f):
                try:
                    db.insert_hand(conn, hand)
                    inserted += 1
                except Exception:
                    skipped += 1
    else:
        print("no fetched phh bulk data found - run scripts/fetch_phh_bulk.py first for real volume")
        for hand in parse_phhs_file(ROOT / "data" / "samples" / "phh" / "fixtures" / "sample_50_hands.phhs"):
            try:
                db.insert_hand(conn, hand)
                inserted += 1
            except Exception:
                skipped += 1

    print(f"inserted {inserted} hands ({skipped} skipped as duplicates) into {DB_PATH}")


if __name__ == "__main__":
    main()
