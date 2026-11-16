# Phase 4 Held-Out Benchmark Report (`hand_pk` Group-Safe Split)

## Split & Training Configuration

- **Dataset source:** `data/poker.db` (`4,992` hands -> `7,442` revealed-card decision points)
- **Partitioning (`poker_analytics.phase4.dataset`):** Deterministic 80/10/10 split **grouped by `hand_pk`** (`seed=42`) so multiple streets (`preflop`, `flop`, `turn`, `river`) from the same hand never cross `train`/`val`/`test`:
  - `train.jsonl`: `6,011` decision points (`train` class counts: `check=2356`, `call=1870`, `bet=1018`, `raise=759`, `fold=8`)
  - `val.jsonl`: `714` decision points
  - `test.jsonl`: `717` decision points
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct` (`496,195,456` total params)
- **LoRA adapter (`phase4/adapters/qwen25_05b_lora`):**
  - Rank `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]` (`2,162,688` trainable params, `0.4359%` of total)
  - **Assistant-only loss masking:** System and user prompt tokens are masked with `-100` in `labels` so cross-entropy loss optimizes only the assistant's action tokens (`fold`, `check`, `call`, `bet <amt>`, `raise to <amt>`).
  - Training slice: `n_train = 2,000`, `n_val = 250`, `epochs = 2.0`, `batch_size = 8`, `lr = 2e-4`
  - Final training loss: `0.3340` | Final validation loss: `0.2746`

## Held-Out Test Benchmark (`n = 250` held-out test decision points)

| Evaluation Arm | Held-Out Test `n` | Correct | Action-Bucket Accuracy | Delta vs. Majority Class |
|---|---:|---:|---:|---:|
| `majority_class_baseline` (`check`) | 250 | 70 | **28.0%** | — |
| `zeroshot_base` (`Qwen/Qwen2.5-0.5B-Instruct`) | 250 | 46 | **18.4%** | -9.6 pp |
| `lora_finetune` (`qwen25_05b_lora`) | 250 | 154 | **61.6%** | **+33.6 pp (+2.20x)** |
