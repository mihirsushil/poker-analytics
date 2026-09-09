# Phase 3 — Tool-Selection Eval Harness

Tests whether an LLM, given a natural-language poker-analysis question and the 6 MCP tools
(`query_hands`, `get_hand`, `compute_stats`, `compare_positions`, `find_leaks`,
`session_summary`) as its available tool set, picks the *correct first tool call* — not whether
its poker analysis is any good. This is a tool-use eval, not a poker-strategy eval.

## Scenario schema

Each YAML file under `scenarios/` is a list of scenarios:

```yaml
- id: query_hands_basic_position          # unique, stable across edits
  category: single_tool                   # single_tool | argument_inference | ambiguous | negative
  prompt: "Show me every hand where I was on the button."
  expected_tool: query_hands              # null for negative cases (no tool should be called)
  expected_args:                          # subset match — only listed keys are checked
    position: BTN
  notes: >
    Straightforward filter-by-position request; tests that "the button" resolves to
    the BTN position code.
```

- **`expected_args` is a subset match.** Only the keys present are checked against the model's
  actual call; unlisted keys (e.g. `limit`, `offset`, exact `player_hash` values that would be
  test-run-specific) are ignored. This avoids over-fitting scenarios to incidental defaults.
- **`category: negative`** scenarios have `expected_tool: null` and `expected_args: {}` — these
  are prompts with no legitimate reason to call any poker tool (small talk, off-topic questions,
  or questions about the *project itself* rather than the data). They measure the model's
  false-positive tool-calling rate, which matters as much as correct-call accuracy: a tool-happy
  model that calls `query_hands` on "what's the weather" is a real failure mode.
- **`category: ambiguous`** scenarios have a defensible correct answer but require resolving
  vague phrasing ("recently," "my biggest leak," "that one session") into concrete arguments.
- **`category: argument_inference`** scenarios test whether natural-language quantities/dates
  ("last month," "since I started 3-betting more") get mapped to the right argument shape, not
  just the right tool.

## Running it

**Not yet run against a live model** — see the harness code in
`src/poker_analytics/eval/` for the scoring/aggregation logic (unit-tested without any API
calls) and `run.py` for the CLI that will drive it once a model/provider decision is made:

```
python -m poker_analytics.eval.run --models claude-sonnet-5,claude-haiku-4-5 --scenarios eval/scenarios
```

Requires `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` in the environment depending on which
`--models` are passed (see `src/poker_analytics/eval/clients.py`).

## Metrics reported

- **Per-tool accuracy** — of the scenarios expecting a given tool, what fraction did the model
  call correctly (right tool + subset-matching args)?
- **False-positive rate** — of the `negative` scenarios, what fraction did the model
  incorrectly call some tool on?
- **Overall accuracy** — correct calls (including correct non-calls on negative scenarios)
  over all scenarios.
