# -*- coding: utf-8 -*-
"""comm_project/src/05_train_sft.py

Day4–Day5 Step 6: SFT training (LoRA/QLoRA) for communication-standards QA.

This version is aligned with the evaluation script (08_evaluate.py) by training
with tokenizer.apply_chat_template (when available).

Inputs:
- comm_project/data/sft/train.jsonl
- comm_project/data/sft/eval.jsonl

Outputs:
- comm_project/outputs/sft_adapter/ (LoRA/QLoRA adapter)
- comm_project/outputs/logs/sft_train.log
- comm_project/outputs/logs/tensorboard/sft/<run_name>/

Key change:
- Training prompt is built with chat_template token ids and labels mask prompt tokens,
  so the model learns to output JSON as the assistant response under chat formatting.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

    ap.add_argument("--tb_logdir", default=str(PROJECT_ROOT / "outputs" / "logs" / "tensorboard" / "sft"))
    ap.add_argument("--run_name", default=None)

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_output_tokens", type=int, default=700, help="Cap output(JSON) length to fit max_length")

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

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--no_4bit", action="store_true")

    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)

    return ap.parse_args()


def detect_use_4bit(args) -> bool:
    if args.no_4bit:
        return False
    if args.use_4bit:
        return True
    try:
        import bitsandbytes as _bnb  # noqa: F401

        return True
    except Exception:
        return False


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_conv(example: dict) -> dict:
    conv = example.get("conversation") or []
    if not conv:
        return {"system": SYSTEM_PROMPT, "input": "", "output": ""}
    item = conv[0]
    return {
        "system": (item.get("system") or SYSTEM_PROMPT).strip(),
        "input": (item.get("input") or "").strip(),
        "output": (item.get("output") or "").strip(),
    }


def build_prompt_ids(tokenizer, system_text: str, user_text: str):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        # Newer transformers can return tensor directly
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            # Some versions return list[int]
            if isinstance(ids, list):
                return ids
            # Some versions return tensor
            if isinstance(ids, torch.Tensor):
                return ids.tolist()[0] if ids.dim() == 2 else ids.tolist()
        except Exception:
            pass

    # Fallback: plain concatenation
    prompt = system_text + "\n" + user_text + "\n"
    return tokenizer(prompt, add_special_tokens=True)["input_ids"]


def main():
    args = parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
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
        run_name = f"{ts}-{model_short}-chatSFT" + ("-4bit" if use_4bit else "")

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

    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if use_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.eval_file})

    def tokenize_fn(ex):
        c = get_conv(ex)
        prompt_ids = build_prompt_ids(tokenizer, c["system"], c["input"])

        # Tokenize output JSON without adding special tokens
        out_ids = tokenizer(c["output"], add_special_tokens=False)["input_ids"]

        # Cap output ids to keep within max_length
        if args.max_output_tokens and len(out_ids) > args.max_output_tokens:
            out_ids = out_ids[: args.max_output_tokens]

        input_ids = prompt_ids + out_ids
        # Truncate from left if too long (keep output)
        if len(input_ids) > args.max_length:
            overflow = len(input_ids) - args.max_length
            # drop from the front of prompt
            prompt_ids_trunc = prompt_ids[overflow:]
            input_ids = prompt_ids_trunc + out_ids
            prompt_ids = prompt_ids_trunc

        labels = [-100] * len(prompt_ids) + out_ids
        labels = labels[-len(input_ids) :]

        attn = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    tokenized = ds.map(tokenize_fn, remove_columns=ds["train"].column_names)

    # Data collator: pad to max in batch
    def collate(features):
        # pad with tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_n)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_n)
            batch["labels"].append(f["labels"] + [-100] * pad_n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

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
        report_to=["tensorboard"],
        logging_dir=str(tb_logdir),
        run_name=run_name,
    )

    from transformers import TrainingArguments as _TA

    try:
        train_args = _TA(**ta_kwargs, eval_strategy="steps", save_strategy="steps")
    except TypeError:
        train_args = _TA(**ta_kwargs, evaluation_strategy="steps", save_strategy="steps")

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collate,
    )

    metrics = trainer.train()

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

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
