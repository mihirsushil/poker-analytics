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

Phase 1 (parser & database) — in progress.

## Phases

1. **Parser & database** — parse hand history text/PHH logs into SQLite;
   VPIP/PFR/3-bet%/etc. stats engine.
2. **MCP server** — expose hand queries and stats to Claude Desktop via MCP.
3. **Eval harness** — measure tool-selection accuracy of models driving the
   MCP server across ambiguous/adversarial scenarios.
4. **LoRA fine-tune** — Qwen2.5-1.5B fine-tuned on decision points to
   predict actions, benchmarked against a prompted frontier model.
