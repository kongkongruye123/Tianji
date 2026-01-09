# -*- coding: utf-8 -*-
"""comm_project/src/08_evaluate.py

Evaluate Base vs SFT vs DPO on a fixed-evidence evaluation set (NO retrieval).

This version aligns with Qwen instruct behavior by using tokenizer.apply_chat_template
when available, and includes robust JSON extraction.

Inputs:
- comm_project/data/eval/eval_set.jsonl
- comm_project/data/corpus/evidence_corpus.jsonl

Outputs:
- reports/eval_<model>.json

Metrics (per plan 11.2):
- json_valid_rate
- schema_pass_rate
- citation_exact_match_rate
- grounded_answer_rate (n-gram coverage proxy)
- refusal_correct_rate
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Local imports
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils.prompts import SYSTEM_PROMPT, format_user_prompt, validate_json_output  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # comm_project/
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.jsonl"
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "evidence_corpus.jsonl"
REPORTS_DIR = PROJECT_ROOT.parent / "reports"

DEFAULT_BASE_MODEL = "qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "outputs" / "sft_adapter"
DEFAULT_DPO_ADAPTER = PROJECT_ROOT / "outputs" / "dpo_adapter"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_evidence_map(corpus_path: Path):
    m = {}
    for row in read_jsonl(corpus_path):
        m[row["chunk_id"]] = row
    return m


def build_evidence_text(chunks):
    blocks = []
    for ch in chunks:
        blocks.append(
            f"[chunk_id={ch['chunk_id']} doc_id={ch['doc_id']} section={ch['section']}]\n{ch['text']}"
        )
    return "\n\n".join(blocks)


def extract_ngrams(text: str, n: int = 3):
    toks = re.findall(r"[A-Za-z0-9_\-]+", (text or "").lower())
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def grounded_coverage(answer: str, evidence_text: str, n: int = 3):
    a = extract_ngrams(answer, n=n)
    if not a:
        return 0.0
    e = extract_ngrams(evidence_text, n=n)
    if not e:
        return 0.0
    hit = sum(1 for g in a if g in e)
    return hit / max(1, len(a))


def extract_json_robust(text: str) -> str:
    """Extract first balanced JSON object from text using brace counting.

    Returns "" if not found.
    """
    if not text:
        return ""
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return ""


@dataclass
class ModelBundle:
    model: object
    tokenizer: object


def load_model_bundle(model_name: str, adapter_path, use_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if use_4bit:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return ModelBundle(model=model, tokenizer=tok)


def build_inputs(bundle: ModelBundle, system_text: str, user_text: str):
    tok = bundle.tokenizer
    messages = [
        {"role": "system", "content": system_text.strip()},
        {"role": "user", "content": user_text.strip()},
    ]

    if hasattr(tok, "apply_chat_template"):
        try:
            input_ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # apply_chat_template may return a tensor of input_ids
            if isinstance(input_ids, torch.Tensor):
                attn = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attn}
        except Exception:
            pass

    # Fallback
    full_prompt = system_text.strip() + "\n" + user_text.strip() + "\n"
    return tok(full_prompt, return_tensors="pt")


def generate_json(bundle: ModelBundle, system_text: str, user_text: str, max_new_tokens: int):
    tok = bundle.tokenizer
    model = bundle.model

    inputs = build_inputs(bundle, system_text, user_text)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    # Decode only newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    return extract_json_robust(gen_text)


def evaluate_one(sample: dict, evidence_map: dict, bundle: ModelBundle, max_new_tokens: int):
    ev_ids = sample["evidence_chunk_ids"]
    chunks = [evidence_map[cid] for cid in ev_ids if cid in evidence_map]
    missing = [cid for cid in ev_ids if cid not in evidence_map]

    evidence_text = build_evidence_text(chunks)
    user_prompt = format_user_prompt(sample["question"], evidence_text=evidence_text)

    raw = generate_json(bundle, SYSTEM_PROMPT, user_prompt, max_new_tokens=max_new_tokens)

    r = {
        "id": sample["id"],
        "type": sample["type"],
        "question": sample["question"],
        "evidence_chunk_ids": ev_ids,
        "missing_evidence": missing,
        "raw_output": raw,
    }

    try:
        obj = json.loads(raw)
        r["json_valid"] = True
        r["parsed"] = obj
    except Exception as e:
        r["json_valid"] = False
        r["parse_error"] = str(e)
        r["schema_pass"] = False
        r["schema_msg"] = "JSON invalid"
        r["citation_total"] = 0
        r["citation_exact_match"] = 0
        r["grounded_coverage"] = 0.0
        r["refusal_correct"] = None
        return r

    ok, msg = validate_json_output(obj)
    r["schema_pass"] = bool(ok)
    r["schema_msg"] = msg

    cit_ok, cit_total = 0, 0
    if isinstance(obj.get("citations"), list):
        for c in obj["citations"]:
            if not isinstance(c, dict):
                continue
            cit_total += 1
            q, cid = c.get("quote"), c.get("chunk_id")
            if isinstance(q, str) and isinstance(cid, str) and cid in evidence_map:
                if q and q in evidence_map[cid].get("text", ""):
                    cit_ok += 1
    r["citation_total"] = cit_total
    r["citation_exact_match"] = cit_ok

    ans = obj.get("answer") if isinstance(obj.get("answer"), str) else ""
    r["grounded_coverage"] = grounded_coverage(ans, evidence_text, n=3)

    if sample["type"] == "unanswerable":
        reason, citations = obj.get("cannot_answer_reason"), obj.get("citations")
        r["refusal_correct"] = bool(isinstance(reason, str) and reason.strip()) and (not citations)
    else:
        r["refusal_correct"] = None

    return r


def aggregate(results):
    total = len(results)
    json_valid = sum(1 for r in results if r.get("json_valid"))
    schema_pass = sum(1 for r in results if r.get("schema_pass"))

    cit_total = sum(r.get("citation_total", 0) for r in results if r.get("json_valid"))
    cit_ok = sum(r.get("citation_exact_match", 0) for r in results if r.get("json_valid"))

    grounded = sum(1 for r in results if r.get("json_valid") and r.get("grounded_coverage", 0) >= 0.2)

    unans = [r for r in results if r.get("type") == "unanswerable" and r.get("json_valid")]
    refusal_ok = sum(1 for r in unans if r.get("refusal_correct"))

    return {
        "total": total,
        "json_valid_rate": json_valid / total if total else 0,
        "schema_pass_rate": schema_pass / total if total else 0,
        "citation_exact_match_rate": cit_ok / cit_total if cit_total else 0,
        "grounded_answer_rate": grounded / total if total else 0,
        "refusal_correct_rate": refusal_ok / len(unans) if unans else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "sft", "dpo"], required=True)
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not EVAL_PATH.exists():
        raise RuntimeError(f"Missing eval set: {EVAL_PATH}")
    if not CORPUS_PATH.exists():
        raise RuntimeError(f"Missing evidence corpus: {CORPUS_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = None
    if args.model == "sft":
        adapter = Path(args.adapter_path) if args.adapter_path else DEFAULT_SFT_ADAPTER
    elif args.model == "dpo":
        adapter = Path(args.adapter_path) if args.adapter_path else DEFAULT_DPO_ADAPTER

    if adapter is not None and not adapter.exists():
        raise RuntimeError(f"Adapter path not found: {adapter}")

    evidence_map = load_evidence_map(CORPUS_PATH)
    samples = list(read_jsonl(EVAL_PATH))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    print(f"[INFO] eval_samples={len(samples)}")
    print(f"[INFO] model={args.model} base_model={args.base_model} adapter={adapter}")

    bundle = load_model_bundle(args.base_model, adapter, use_4bit=args.use_4bit)

    results = []
    for s in samples:
        results.append(evaluate_one(s, evidence_map, bundle, max_new_tokens=args.max_new_tokens))
        if len(results) % 10 == 0:
            print(f"  done {len(results)}/{len(samples)}")

    summary = aggregate(results)
    out_path = REPORTS_DIR / f"eval_{args.model}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] wrote report: {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
