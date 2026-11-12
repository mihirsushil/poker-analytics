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

Phase 2 (MCP server) is done. Phase 1 (parser & database) core is done and validated:

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

Phase 2 (MCP server) exposes six tools over the database in
`src/poker_analytics/mcp_server.py`: `query_hands`, `get_hand`,
`compute_stats`, `compare_positions`, `find_leaks`, `session_summary`.

- `find_leaks` flags a player's stats that deviate from the population
  baseline by more than a per-stat threshold (documented in
  `stats._LEAK_THRESHOLDS`), gated by a minimum sample size so a handful of
  hands can't trigger a false positive.
- `session_summary` buckets a player's hands into sessions using a
  gap-based rule (default 30 minutes, configurable), the same approach HUD
  tools like PT4/HM3 use - multi-tabling counts as one session, not one
  per table. Note: the bootstrap PHH bulk data only carries date-level
  timestamps upstream (no time-of-day), so session detection on it
  degenerates to one bucket per day; it's fully meaningful on the raw-text
  fixtures and will be on self-played data, which carry real timestamps.

Run it: `python -m poker_analytics.mcp_server` (or the `poker-analytics-mcp`
console script after `pip install -e .`), registered with an MCP client
over stdio.

Phase 3 (MCP tool-selection evaluation harness) lives in
`src/poker_analytics/eval/` (`poker-analytics-eval`): 30 declarative YAML
scenarios in `eval/scenarios.yaml` (single-tool, multi-step, and no-tool/guardrail
cases) with deterministic argument-shape grading in `eval/grading.py` and cost/latency
accounting in `eval/report.py`. Also serves as the benchmark target for
[`mcp-eval-toolkit`](https://github.com/mihirsushil/mcp-eval-toolkit).

Phase 4 (decision-point extraction, group-safe dataset split, and LoRA fine-tuning
pipeline) lives in `src/poker_analytics/phase4/` (see [`phase4/README.md`](phase4/README.md)
for details):

- `poker-analytics-extract` walks `hands`/`hand_seats`/`actions`/`pots` to emit
  supervised `(state, action)` decision points for seats with known hole cards.
- `poker-analytics-dataset` splits decision points into `train`/`val`/`test`
  **grouped by `hand_pk`** so multiple streets from the same hand never leak across
  partitions, and formats chat-template JSONL.
- `poker-analytics-finetune` and `poker-analytics-benchmark` run Qwen2.5 LoRA
  fine-tuning (`peft`) and compare against a `MajorityClassBaseline` and an optional
  prompted frontier model.

## Reproducible setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval,phase4]"
pytest -v
```

## Phases

1. **Parser & database** — parse hand history text/PHH logs into SQLite;
   VPIP/PFR/3-bet%/etc. stats engine.
2. **MCP server** — expose hand queries and stats to Claude Desktop via MCP.
3. **Eval harness** — measure tool-selection accuracy of models driving the
   MCP server across ambiguous/adversarial scenarios.
4. **LoRA fine-tune** — Qwen2.5-1.5B fine-tuned on decision points to
   predict actions, benchmarked against a prompted frontier model.
