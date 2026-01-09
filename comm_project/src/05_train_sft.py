# -*- coding: utf-8 -*-
"""comm_project/src/05_train_sft.py

Day4–Day5 Step 6: SFT training (LoRA/QLoRA) for communication-standards QA.

This script is a comm_project-oriented wrapper that follows docs/comm_llm_plan.md.

Inputs:
- comm_project/data/sft/train.jsonl
- comm_project/data/sft/eval.jsonl

Outputs:
- comm_project/outputs/sft_adapter/ (LoRA/QLoRA adapter)
- comm_project/outputs/logs/sft_train.log
- comm_project/outputs/logs/tensorboard/sft/<run_name>/  (TensorBoard events)

Key constraints from plan:
- Model: qwen/Qwen2.5-1.5B-Instruct (default)
- Prefer QLoRA 4-bit when possible; fallback to LoRA bf16/fp16
- Max length: 2048 (fallback 1024)

Dataset format expected (from 04_make_sft.py):
{"conversation": [{"system":..., "input":..., "output":...}]}

We train causal LM to generate output given (system+input) as prompt.

TensorBoard visualization:
- This script logs training/eval metrics to TensorBoard.
- Start TensorBoard after (or during) training:
    tensorboard --logdir comm_project/outputs/logs/tensorboard

Notes:
- This script requires: transformers, datasets, peft, accelerate
- Optional (for QLoRA): bitsandbytes
- Optional (for TensorBoard UI): tensorboard
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # comm_project/

# Ensure we can import comm_project/src/utils/prompts.py when running from repo root
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils.prompts import SYSTEM_PROMPT  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", default="qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_file", default=str(PROJECT_ROOT / "data" / "sft" / "train.jsonl"))
    ap.add_argument("--eval_file", default=str(PROJECT_ROOT / "data" / "sft" / "eval.jsonl"))
    ap.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "sft_adapter"))
    ap.add_argument("--log_file", default=str(PROJECT_ROOT / "outputs" / "logs" / "sft_train.log"))

    # TensorBoard
    ap.add_argument(
        "--tb_logdir",
        default=str(PROJECT_ROOT / "outputs" / "logs" / "tensorboard" / "sft"),
        help="TensorBoard log directory root",
    )
    ap.add_argument(
        "--run_name",
        default=None,
        help="TensorBoard run name (default: auto from timestamp + model)",
    )

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=1000)

    ap.add_argument("--seed", type=int, default=42)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # QLoRA
    ap.add_argument("--use_4bit", action="store_true", help="Force enable 4bit QLoRA")
    ap.add_argument("--no_4bit", action="store_true", help="Force disable 4bit and use bf16/fp16 LoRA")

    # Memory
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)

    return ap.parse_args()


def detect_use_4bit(args) -> bool:
    if args.no_4bit:
        return False
    if args.use_4bit:
        return True
    # Auto: enable 4bit if bitsandbytes is available
    try:
        import bitsandbytes as _bnb  # noqa: F401

        return True
    except Exception:
        return False


def get_dtype():
    # Prefer bf16 if available
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_prompt(example: dict) -> str:
    """Build training prompt string from a single dataset example."""
    conv = example.get("conversation") or []
    if not conv:
        return ""
    item = conv[0]
    sys_text = item.get("system") or SYSTEM_PROMPT
    inp = item.get("input") or ""
    return sys_text.strip() + "\n" + inp.strip() + "\n"


def build_output(example: dict) -> str:
    conv = example.get("conversation") or []
    if not conv:
        return ""
    return (conv[0].get("output") or "").strip()


def main():
    args = parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from peft import LoraConfig, get_peft_model

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    use_4bit = detect_use_4bit(args)
    dtype = get_dtype()

    # TensorBoard run name
    if args.run_name:
        run_name = args.run_name
    else:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        model_short = args.model_name.split("/")[-1]
        run_name = f"{ts}-{model_short}-lora"
        run_name += "-4bit" if use_4bit else ""

    tb_logdir = Path(args.tb_logdir) / run_name
    tb_logdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] model_name={args.model_name}")
    print(f"[INFO] train_file={args.train_file}")
    print(f"[INFO] eval_file={args.eval_file}")
    print(f"[INFO] output_dir={args.output_dir}")
    print(f"[INFO] use_4bit={use_4bit} dtype={dtype}")
    print(f"[INFO] tensorboard_logdir={tb_logdir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {"trust_remote_code": True}
    if use_4bit:
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "torch_dtype": dtype,
                "device_map": "auto",
            }
        )
    else:
        model_kwargs.update(
            {
                "torch_dtype": dtype,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # LoRA config (Qwen2.5 typical attention proj names)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    data_files = {"train": args.train_file, "validation": args.eval_file}
    ds = load_dataset("json", data_files=data_files)

    def tokenize_fn(ex):
        prompt = build_prompt(ex)
        output = build_output(ex)
        full = prompt + output

        enc = tokenizer(
            full,
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )

        prompt_ids = tokenizer(
            prompt,
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )["input_ids"]

        labels = enc["input_ids"].copy()
        prompt_len = len(prompt_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        enc["labels"] = labels
        return enc

    tokenized = ds.map(tokenize_fn, remove_columns=ds["train"].column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # transformers TrainingArguments API changed across versions.
    # Some versions use `eval_strategy` instead of `evaluation_strategy`.
    ta_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        # TensorBoard
        report_to=["tensorboard"],
        logging_dir=str(tb_logdir),
        run_name=run_name,
    )

    # Try newer argument name first, fallback to older.
    try:
        train_args = TrainingArguments(
            **ta_kwargs,
            eval_strategy="steps",
            save_strategy="steps",
        )
    except TypeError:
        train_args = TrainingArguments(
            **ta_kwargs,
            evaluation_strategy="steps",
            save_strategy="steps",
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    metrics = trainer.train()

    # Save adapter
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Write log
    log_path = Path(args.log_file)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"trained_at={datetime.utcnow().isoformat()}Z\n")
        f.write(f"run_name={run_name}\n")
        f.write(f"tensorboard_logdir={tb_logdir}\n")
        f.write(json.dumps({"train_metrics": metrics.metrics}, ensure_ascii=False) + "\n")

    print(f"[OK] saved adapter to: {output_dir}")
    print(f"[OK] wrote log to: {log_path}")
    print(f"[OK] tensorboard logdir: {tb_logdir}")


if __name__ == "__main__":
    main()
