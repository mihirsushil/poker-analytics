"""Phase 4 — LoRA fine-tune of a small LLM on real poker decision points,
benchmarked against a majority-class baseline and a prompted frontier model.

Submodules:
- extract: walk the Phase 1 SQLite schema (hands/hand_seats/actions/pots)
  and emit one DecisionExample per action taken by a seat whose hole cards
  are known.
- dataset: split extracted examples into train/val/test and format them as
  instruction/response JSONL for causal-LM fine-tuning.
- finetune: LoRA fine-tune script (peft + transformers).
- infer: load a base model + LoRA adapter and generate an action prediction.
- baseline: majority-class baseline.
- frontier: prompted-frontier-model stub, gated on an API key being set.
- benchmark: run all three and report decision-prediction accuracy.
"""
