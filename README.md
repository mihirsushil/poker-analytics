# Poker Analytics Platform & Action Prediction Model

A four-phase project built on poker hand histories: parser & database → MCP
server → eval harness → LoRA fine-tuned action-prediction model.

## What this is not

This is **not** a poker bot. Nothing here plays poker automatically or in
real time — every tool operates on historical hand data after the fact, for
retrospective strategy analysis. Automated play violates the terms of
service of every online poker site and is explicitly out of scope.

## Data & privacy

- Opponent screen names are hashed at parse time, before anything is stored
  or queried. Anonymization happens in the parser, not as an afterthought.
- **Bootstrap phase:** this repo currently ships on public, third-party hand
  history data (see `data/samples/README.md` for exact sources and
  licenses) so that the parser, MCP server, and eval harness can be built
  and demonstrated before enough self-played volume exists.
- **Long-term:** the intent is to replace/supplement the bootstrap data with
  the author's own played hands. Any self-played data added later will be
  called out explicitly; nothing here is presented as self-played unless it
  is.

## Status

Phase 1 (parser & database) core is done and validated:

- Two parsers feeding one schema: raw PokerStars `.txt` hand histories
  (validated against 15 real fixture hands covering side pots, all-ins,
  uncalled bets, sitting-out players, and both header format eras) and the
  public `phh-dataset` bulk-volume source.
- **4,992 hands** ingested into `data/poker.db` (run
  `scripts/fetch_phh_bulk.py` then `scripts/ingest_bootstrap_data.py` to
  reproduce/expand).
- Stats engine (VPIP, PFR, 3-bet%, fold-to-3-bet%, c-bet%, WTSD%, W$SD%,
  aggression factor), sliceable by player/position/date. VPIP and PFR are
  cross-checked against an independent brute-force recomputation
  (`tests/test_stats.py`), per the project's own validation bar.
- Sanity check on the current dataset: VPIP/PFR rise from UTG (20.4%/7.5%)
  to BTN (29.7%/14.3%), SB VPIP is the highest of any position (35.0%), and
  W$SD sits at 50.8% - all exactly the shape you'd expect from a real
  population, which is the best available substitute for PokerTracker
  cross-validation right now.

Next: Phase 2 (MCP server).

## Phases

1. **Parser & database** — parse hand history text/PHH logs into SQLite;
   VPIP/PFR/3-bet%/etc. stats engine.
2. **MCP server** — expose hand queries and stats to Claude Desktop via MCP.
3. **Eval harness** — measure tool-selection accuracy of models driving the
   MCP server across ambiguous/adversarial scenarios.
4. **LoRA fine-tune** — Qwen2.5-1.5B fine-tuned on decision points to
   predict actions, benchmarked against a prompted frontier model.
