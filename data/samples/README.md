# Bootstrap sample data (not self-played)

This project starts on public, third-party data so Phases 1-3 can be built
and demonstrated before enough self-played hand volume exists. None of this
is presented as the author's own play. See the root `README.md` for the
project's data/privacy stance.

## `raw_text/`

15 real PokerStars hand-history `.txt` exports, vendored from
[thlorenz/hhp](https://github.com/thlorenz/hhp) (MIT license), from
`test/fixtures/holdem/pokerstars/`. Chosen specifically because they cover
the hard parsing edge cases called out in the project plan:

- `allin-preflop.txt`, `call-allin.txt`, `posts-allin.txt` — all-ins
- `collect-on-flop.txt`, `collected-on-showdown.txt`, `uncalled-bet-returned.txt` — pot/side-pot mechanics
- `player-sitting-out.txt` — disconnect/sit-out handling
- `cash.zoom.2017.muck-cards.txt`, `cash.zoom.2017.parens-in-playername.txt` — text-parsing edge cases (mucked hands, punctuation in screen names)
- `2008-tournament.txt`, `cash.2010.txt` — older PokerStars header format (`PokerStars Game #...`) vs. the modern format (`PokerStars Hand #...`) — the parser needs to handle both
- `timezone-costarica.sng.txt`, `freeroll.tny.txt`, `actiononall.txt` — tournament/SNG variants

Small on purpose (one hand per file): this set is for validating parser
correctness against real edge cases, not for volume.

## `phh/`

Bulk volume comes from [uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset)
(MIT license) — real, already-anonymized 2009 online-poker hand histories
(the HandHQ dataset, ~3M PokerStars hands among ~21.6M total across several
sites) in the [PHH format](https://phh.readthedocs.io/en/stable/), an
open, structured hand-history spec. Fetched on demand via
`scripts/fetch_phh_bulk.py` rather than committed to the repo (see
`.gitignore`) since it's large and easily re-fetched.

These hands are pre-anonymized upstream (player identifiers are already
opaque tokens), so they need no additional hashing — but the project's own
hashing step still runs on them, same as it would on self-played data.
