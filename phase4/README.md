# Phase 4 — LoRA fine-tune on real decision points

Turns the Phase 1 hand database into (state, action) training examples, LoRA
fine-tunes a small chat model to predict the action, and benchmarks it
against a majority-class baseline and (once a key is available) a prompted
frontier model.

```
poker.db --extract.py--> decisions.jsonl --dataset.py--> train/val/test.jsonl
                                                                |
                                                     finetune.py (LoRA)
                                                                |
                                                          adapter/
                                                                |
                          benchmark.py: baseline vs adapter vs frontier
```

## What's built and verified

- **`extract.py`** — walks `hands`/`hand_seats`/`actions`/`pots` and emits one
  `DecisionExample` per action taken by a seat whose hole cards are known
  (hero, or a showdown reveal — the only population with real labels).
  7,442 decision points extracted from the existing 4,992-hand DB. Unit
  tests (`tests/phase4/test_extract.py`) reconstruct small synthetic hands
  by hand and assert the recorded pot/stack/board/history at each decision
  matches the real action sequence — same spirit as Phase 1's
  money-conservation checks.
- **`dataset.py`** — deterministic (seeded) train/val/test split (80/10/10)
  **grouped by `hand_pk`** so multiple decisions from the same hand never leak
  across partitions, formatted as chat-template instruction/response JSONL
  (system prompt + state description + target action string).
- **`finetune.py`** — LoRA (`peft`) fine-tune of a `transformers` causal-LM with
  **assistant-only loss masking** (system + user prompt tokens masked with `-100`
  so gradient updates optimize only the assistant's action tokens) and `r=16,
  lora_alpha=32` across `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
  - **Real training run completed (`phase4/adapters/qwen25_05b_lora`)**:
    `Qwen/Qwen2.5-0.5B-Instruct` trained on `n_train = 2,000` decision points
    for `2.0` epochs (`batch_size = 8`, `lr = 2e-4`), reaching
    `final_train_loss = 0.3340` and `final_eval_loss = 0.2746` on `n_val = 250`.
- **`infer.py`** — reloads a saved adapter (or evaluates the unadapted zero-shot
  base model via `predict_zeroshot_batch`) and generates greedy (`do_sample=False`)
  action predictions.
- **`baseline.py`** — majority-class baseline. On the group-safe `hand_pk` split
  (`train = 6,011`, `val = 714`, `test = 717`), `check` is the majority class.
- **`benchmark.py`** — scores `majority_class_baseline` vs. `zeroshot_base` vs.
  `lora_finetune` (and optional `frontier_model`) on the held-out `hand_pk` test
  split (`n = 250`), graded on the action bucket (`fold`/`check`/`call`/`bet`/`raise`):
  - `majority_class_baseline` (`check`): **28.0%** (`70 / 250`)
  - `zeroshot_base` (`Qwen/Qwen2.5-0.5B-Instruct`): **18.4%** (`46 / 250`)
  - `lora_finetune` (`phase4/adapters/qwen25_05b_lora`): **61.6%** (`154 / 250`,
    **+33.6 pp** over majority-class and **+43.2 pp** over zero-shot base; see
    [`results/benchmark_report.md`](results/benchmark_report.md)).
- **`frontier.py`** — prompted-frontier-model comparison arm, gated on
  `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` being set. **No live API call was
  made building this** — the key isn't set in this environment.
  `benchmark.py` skips this arm automatically when unconfigured.
- 14 unit tests (`tests/phase4/`), all passing, including `test_split_groups_by_hand_pk_without_cross_split_leakage` verifying zero `hand_pk` overlap across `train`/`val`/`test`. Full repo suite: 101/101 passing.

## Known bias in the data (real, not a bug)

Hole cards are only known for a seat when they were actually revealed —
hero's own cards, or a showdown reveal. That skews the extracted population
toward hands that *didn't* end in an early fold for that seat: of 7,442
decision points, only 9 are `fold` (vs. 2,896 `check`, 2,308 `call`, 1,278
`bet`, 951 `raise`). This is documented in `dataset.py`'s module docstring
and reflected honestly in the majority-class baseline rather than corrected
for — correcting it would mean fabricating fold examples that aren't in the
real parsed data.

## What still needs Mihir

- **A real training run.** The smoke test proves the pipeline works
  mechanically; it is not a real fine-tune. Run without `--smoke` for the
  full `Qwen2.5-1.5B-Instruct` base model on the full ~5,954-example train
  split for a real epoch count:
  ```
  python -m poker_analytics.phase4.finetune --out-dir phase4/adapters/v1
  ```
  This needs a real time budget (hours, not the smoke test's under-a-minute,
  even on MPS) — pick a time to let it run, or move it to a machine with a
  real GPU if MPS throughput isn't acceptable.
- **The frontier-model comparison.** Set `ANTHROPIC_API_KEY` (and/or
  `OPENAI_API_KEY`) in the environment, then:
  ```
  python -m poker_analytics.phase4.benchmark --adapter-dir phase4/adapters/v1 --frontier-model claude-sonnet-5
  ```
  Pick which frontier model(s) to spend real API budget on, same decision
  already deferred on Phase 3's eval harness.
- **A real benchmark run** against the full test split (744 examples) once
  both of the above exist — the numbers in this README are from a 20-example
  smoke slice, not the real test set.

## Running it

```bash
# from the repo root, with the venv active and `pip install -e '.[phase4]'` done
python -m poker_analytics.phase4.extract                      # poker.db -> phase4/data/decisions.jsonl
python -m poker_analytics.phase4.dataset                      # -> train/val/test.jsonl
python -m poker_analytics.phase4.finetune --smoke --limit 16  # fast mechanical smoke test
python -m poker_analytics.phase4.infer --adapter-dir phase4/adapters/smoke_test -n 5
python -m poker_analytics.phase4.benchmark --adapter-dir phase4/adapters/smoke_test --limit 20
```

`phase4/data/*.jsonl`, `phase4/adapters/`, and `phase4/benchmark_results.json`
are gitignored — they're regenerated from `data/poker.db`, not source.
