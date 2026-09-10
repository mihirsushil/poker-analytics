"""LoRA fine-tune of a small causal-LM chat model on poker decision points.

Default base model is Qwen2.5-1.5B-Instruct, the real Phase 4 target. For a
fast local smoke test (proving the pipeline loads a model, applies LoRA,
trains, saves an adapter, and can reload+generate - not proving the model
has actually learned good poker strategy) pass --smoke, which swaps in
Qwen2.5-0.5B-Instruct and trains on a small --limit slice for one epoch.
The full run (no --smoke, full train.jsonl, more epochs) needs a real time
budget and is left for Mihir to kick off - see phase4/README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FULL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_chat_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_dataset(rows: list[dict], tokenizer, max_length: int = 512):
    from datasets import Dataset

    def _to_text(row):
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)

    texts = [_to_text(r) for r in rows]
    ds = Dataset.from_dict({"text": texts})

    def _tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")
        # Mask pad positions so the loss isn't dominated by padding tokens.
        out["labels"] = [
            [tok if tok != tokenizer.pad_token_id else -100 for tok in ids]
            for ids in out["input_ids"]
        ]
        return out

    return ds.map(_tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a base model on poker decision points.")
    parser.add_argument("--train", default="phase4/data/train.jsonl")
    parser.add_argument("--val", default="phase4/data/val.jsonl")
    parser.add_argument("--out-dir", default="phase4/adapters/latest")
    parser.add_argument("--model", default=None, help="override base model id")
    parser.add_argument("--smoke", action="store_true", help="fast smoke test: small model, small slice, 1 epoch")
    parser.add_argument("--limit", type=int, default=None, help="cap number of training examples")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if args.smoke:
        model_id = args.model or SMOKE_MODEL
        limit = args.limit or 64
        epochs = args.epochs or 1
    else:
        model_id = args.model or FULL_MODEL
        limit = args.limit
        epochs = args.epochs or 3

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_id} on {device} (smoke={args.smoke})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_chat_jsonl(args.train)
    val_rows = load_chat_jsonl(args.val)
    if limit:
        train_rows = train_rows[:limit]
        val_rows = val_rows[: max(1, limit // 8)]

    train_ds = build_dataset(train_rows, tokenizer)
    val_ds = build_dataset(val_rows, tokenizer)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        learning_rate=2e-4,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    result = trainer.train()
    eval_result = trainer.evaluate()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "base_model": model_id,
        "smoke": args.smoke,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "epochs": epochs,
        "final_train_loss": result.training_loss,
        "final_eval_loss": eval_result.get("eval_loss"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
