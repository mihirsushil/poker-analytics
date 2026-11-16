"""Load a base model + LoRA adapter and generate an action prediction for a
decision point. Used by benchmark.py; also runnable standalone to sanity
check a trained adapter actually produces text.
"""
from __future__ import annotations

from functools import lru_cache

from .dataset import SYSTEM_PROMPT, format_prompt
from .extract import DecisionExample


@lru_cache(maxsize=2)
def _load(adapter_dir: str):
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta = json.loads((__import__("pathlib").Path(adapter_dir) / "run_meta.json").read_text())
    base_model_id = meta["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


@lru_cache(maxsize=2)
def _load_base(base_model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    base.to(device)
    base.eval()
    return base, tokenizer, device


def predict_action_base(ex: DecisionExample, base_model_id: str, max_new_tokens: int = 12) -> str:
    import torch

    model, tokenizer, device = _load_base(base_model_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(ex)},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def predict_action(ex: DecisionExample, adapter_dir: str, max_new_tokens: int = 12) -> str:
    import torch

    model, tokenizer, device = _load(adapter_dir)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(ex)},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    import argparse

    from .dataset import load_examples

    parser = argparse.ArgumentParser(description="Sanity-check a trained adapter by generating a few predictions.")
    parser.add_argument("--adapter-dir", default="phase4/adapters/smoke_test")
    parser.add_argument("--examples", default="phase4/data/test.jsonl", help="chat JSONL is fine too")
    parser.add_argument("--decisions", default="phase4/data/decisions.jsonl", help="raw decisions.jsonl for real DecisionExample fields")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    examples = load_examples(args.decisions)[: args.n]
    for ex in examples:
        pred = predict_action(ex, args.adapter_dir)
        print(f"actual={ex.action_label!r:20} predicted={pred!r}")


if __name__ == "__main__":
    main()
