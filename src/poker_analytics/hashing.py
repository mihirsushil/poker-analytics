"""Stable, salted hashing for opponent screen names.

Anonymization happens at parse time, before anything reaches the database,
per the project's data/privacy rules. Hashes are salted so they aren't
directly reversible via lookup tables of common screen names, but remain
stable across parses (same name -> same hash) so per-player stats still
work.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_SALT_FILE = Path(__file__).resolve().parent.parent.parent / "data" / ".hash_salt"


def _load_or_create_salt() -> str:
    if _SALT_FILE.exists():
        return _SALT_FILE.read_text().strip()
    salt = os.urandom(32).hex()
    _SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SALT_FILE.write_text(salt)
    return salt


def hash_player_name(name: str, salt: str | None = None) -> str:
    """Return a stable, salted hash for a screen name.

    Truncated to 16 hex chars (64 bits) - plenty of collision resistance for
    a personal hand-history dataset, and short enough to stay readable in
    query results.
    """
    salt = salt if salt is not None else _load_or_create_salt()
    digest = hashlib.sha256(f"{salt}:{name}".encode("utf-8")).hexdigest()
    return digest[:16]
