# -*- coding: utf-8 -*-
"""comm_project/src/06_make_dpo.py

Day6 Step 7: Build DPO preference pairs from fixed-evidence prompts.

Goal
- Produce (prompt, chosen, rejected) pairs that improve:
  - JSON validity / schema stability
  - citation quote exact-match
  - refusal correctness when evidence is insufficient

Strategy (per docs/comm_llm_plan.md 9.4)
- For each prompt, generate two candidate responses A/B using the same model but different decoding
  settings (temperature 0.2 vs 0.8). Optionally you can compare base vs sft.
- Score candidates with deterministic rules (schema, quote substring, refusal correctness, groundedness proxy).
- Keep pairs only when score gap >= --min_margin.

Inputs
- comm_project/data/eval/eval_set.jsonl (fixed evidence ids)
- comm_project/data/corpus/evidence_corpus.jsonl (chunk texts)

Outputs
- comm_project/data/dpo/train.jsonl
- comm_project/data/dpo/eval.jsonl
- comm_project/data/dpo/quality_report.json

JSONL output format (TRL DPO friendly)
Each line:
{
  "prompt": "<full prompt text>",
  "chosen": "<assistant output JSON>",
  "rejected": "<assistant output JSON>",
  "meta": {...}
}

Incremental writing
- When --incremental is enabled, the script appends pairs to disk during generation.
- This allows you to interrupt and still keep already generated data.
- With --resume, the script will continue from existing output files by counting existing lines.

Notes
- We store the assistant raw JSON string as chosen/rejected (not wrapped in ```).
- prompt is stored as plain text concatenation (system + user) for portability.
"""

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils.prompts import SYSTEM_PROMPT, format_user_prompt, validate_json_output  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.jsonl"
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "evidence_corpus.jsonl"
DPO_DIR = PROJECT_ROOT / "data" / "dpo"
TRAIN_OUT = DPO_DIR / "train.jsonl"
EVAL_OUT = DPO_DIR / "eval.jsonl"
REPORT_OUT = DPO_DIR / "quality_report.json"

DEFAULT_BASE_MODEL = "qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "outputs" / "sft_adapter"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_evidence_map() -> Dict[str, dict]:
    m = {}
    for row in read_jsonl(CORPUS_PATH):
        m[row["chunk_id"]] = row
    return m


def build_evidence_text(chunks: List[dict]) -> str:
    blocks = []
    for ch in chunks:
        blocks.append(
            f"[chunk_id={ch['chunk_id']} doc_id={ch['doc_id']} section={ch['section']}]\n{ch['text']}"
        )
    return "\n\n".join(blocks)


def extract_json_robust(text: str) -> str:
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


def extract_ngrams(text: str, n: int = 3) -> set:
    toks = re.findall(r"[A-Za-z0-9_\-]+", (text or "").lower())
    if len(toks) < n:
        return set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def grounded_coverage(answer: str, evidence_text: str, n: int = 3) -> float:
    a = extract_ngrams(answer, n=n)
    if not a:
        return 0.0
    e = extract_ngrams(evidence_text, n=n)
    if not e:
        return 0.0
    hit = sum(1 for g in a if g in e)
    return hit / max(1, len(a))


@dataclass
class ModelBundle:
    model: object
    tokenizer: object


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_bundle(model_name: str, adapter_path: Optional[Path], use_4bit: bool) -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = get_dtype()

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


def build_full_prompt(system_text: str, user_text: str) -> str:
    return system_text.strip() + "\n\n" + user_text.strip() + "\n"


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
            if isinstance(input_ids, torch.Tensor):
                attn = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attn}
        except Exception:
            pass

    full_prompt = build_full_prompt(system_text, user_text)
    return tok(full_prompt, return_tensors="pt")


def generate_one(
    bundle: ModelBundle,
    system_text: str,
    user_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    tok = bundle.tokenizer
    model = bundle.model

    torch.manual_seed(seed)

    inputs = build_inputs(bundle, system_text, user_text)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tok.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    return extract_json_robust(gen_text)


def score_candidate(
    sample_type: str,
    candidate_json_str: str,
    evidence_map: Dict[str, dict],
    evidence_text: str,
) -> Tuple[int, Dict[str, float]]:
    features: Dict[str, float] = {}

    if not candidate_json_str:
        return -10, {"empty": 1}

    try:
        obj = json.loads(candidate_json_str)
        features["json_parse"] = 1
    except Exception:
        return -8, {"json_parse": 0}

    score = 2  # json parse

    ok, _msg = validate_json_output(obj)
    features["schema_ok"] = 1 if ok else 0
    score += 2 if ok else -2

    cannot = obj.get("cannot_answer_reason")
    cits = obj.get("citations") or []
    ans = obj.get("answer") or ""

    if sample_type == "unanswerable":
        if isinstance(cannot, str) and cannot.strip():
            score += 2
            features["refusal_reason"] = 1
        else:
            score -= 3
            features["refusal_reason"] = 0

        if not cits:
            score += 1
            features["refusal_empty_cits"] = 1
        else:
            score -= 2
            features["refusal_empty_cits"] = 0

        bad_quote = 0
        for cit in cits:
            cid = cit.get("chunk_id")
            qt = cit.get("quote")
            ev = evidence_map.get(cid)
            if ev and isinstance(qt, str) and qt and qt in ev.get("text", ""):
                continue
            bad_quote += 1
        if bad_quote:
            score -= 5
            features["bad_quote"] = bad_quote

        return score, features

    # answerable
    if cannot is None:
        features["cannot_null"] = 1
    else:
        score -= 4
        features["cannot_null"] = 0

    if cits:
        score += 1
        features["has_citation"] = 1
    else:
        score -= 4
        features["has_citation"] = 0

    quote_ok = 0
    bad_quote = 0
    for cit in cits[:3]:
        cid = cit.get("chunk_id")
        qt = cit.get("quote")
        ev = evidence_map.get(cid)
        if ev and isinstance(qt, str) and qt and qt in ev.get("text", ""):
            quote_ok += 1
        else:
            bad_quote += 1

    features["quote_ok"] = quote_ok
    features["bad_quote"] = bad_quote
    score += 3 * quote_ok
    score -= 5 * bad_quote

    cov = grounded_coverage(ans, evidence_text, n=3)
    features["grounded_cov"] = cov
    if cov >= 0.3:
        score += 2
    elif cov >= 0.15:
        score += 1
    else:
        score -= 2

    return score, features


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", default=DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--adapter_path",
        default=str(DEFAULT_SFT_ADAPTER),
        help="Use SFT adapter to generate better candidates",
    )

    ap.add_argument("--use_4bit", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=512)

    ap.add_argument("--temperature_a", type=float, default=0.2)
    ap.add_argument("--temperature_b", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_pairs", type=int, default=6000)
    ap.add_argument("--eval_size", type=int, default=500)
    ap.add_argument("--min_margin", type=int, default=3)

    ap.add_argument("--out_train", default=str(TRAIN_OUT))
    ap.add_argument("--out_eval", default=str(EVAL_OUT))

    ap.add_argument("--incremental", action="store_true", help="Append pairs to disk during generation")
    ap.add_argument("--resume", action="store_true", help="Resume by counting existing output lines and appending")
    ap.add_argument("--flush_every", type=int, default=10, help="Flush progress to disk every N kept pairs")
    ap.add_argument("--progress_every", type=int, default=50, help="Print progress every N tries")

    args = ap.parse_args()

    rng = random.Random(args.seed)

    if not EVAL_PATH.exists():
        raise RuntimeError(f"Missing eval set: {EVAL_PATH}. Run 03_make_eval.py first.")
    if not CORPUS_PATH.exists():
        raise RuntimeError(f"Missing corpus: {CORPUS_PATH}. Run 02_build_corpus.py first.")

    evidence_map = load_evidence_map()
    eval_set = list(read_jsonl(EVAL_PATH))
    rng.shuffle(eval_set)

    adapter = Path(args.adapter_path) if args.adapter_path and args.adapter_path.lower() != "none" else None

    print(f"[INFO] loading model bundle: model_name={args.model_name} adapter={adapter} use_4bit={args.use_4bit}")
    bundle = load_model_bundle(args.model_name, adapter, use_4bit=args.use_4bit)
    print("[INFO] model bundle loaded")

    out_train = Path(args.out_train)
    out_eval = Path(args.out_eval)

    # Incremental mode layout: keep eval first, then train.
    # If resuming, infer how many already written.
    existing_eval = count_jsonl_lines(out_eval) if args.resume else 0
    existing_train = count_jsonl_lines(out_train) if args.resume else 0

    target_eval = min(args.eval_size, max(1, args.num_pairs // 10))
    target_train = max(0, args.num_pairs - target_eval)

    if args.incremental and not args.resume:
        # Start fresh (overwrite)
        out_eval.parent.mkdir(parents=True, exist_ok=True)
        out_eval.write_text("", encoding="utf-8")
        out_train.write_text("", encoding="utf-8")
        existing_eval = 0
        existing_train = 0

    kept_eval = existing_eval
    kept_train = existing_train
    kept_total = kept_eval + kept_train

    print(
        f"[INFO] target pairs: total={args.num_pairs} eval={target_eval} train={target_train}; "
        f"resume={args.resume} incremental={args.incremental}; existing: eval={existing_eval} train={existing_train}",
        flush=True,
    )

    import time

    t0 = time.time()
    tried = 0
    buffer = []

    def flush_buffer():
        nonlocal buffer
        if not buffer:
            return
        for row, split in buffer:
            if split == "eval":
                append_jsonl(out_eval, row)
            else:
                append_jsonl(out_train, row)
        buffer = []

    # Sample prompts with replacement
    while kept_total < args.num_pairs and tried < args.num_pairs * 8:
        tried += 1

        if args.progress_every > 0 and tried % args.progress_every == 0:
            elapsed = time.time() - t0
            keep_rate = kept_total / tried if tried else 0.0
            eta = ((args.num_pairs - kept_total) / max(1e-9, (kept_total / elapsed))) if kept_total > 0 else float("inf")
            eta_str = f"{eta/60:.1f} min" if eta != float("inf") else "inf"
            print(
                f"[PROGRESS] tried={tried} kept={kept_total}/{args.num_pairs} keep_rate={keep_rate:.3f} elapsed={elapsed/60:.1f} min eta≈{eta_str}",
                flush=True,
            )

        s = rng.choice(eval_set)
        ev_ids = s.get("evidence_chunk_ids") or []
        chunks = [evidence_map[cid] for cid in ev_ids if cid in evidence_map]
        if not chunks:
            continue

        evidence_text = build_evidence_text(chunks)
        user_prompt = format_user_prompt(s["question"], evidence_text=evidence_text)

        base_seed = rng.randrange(0, 2**31 - 1)

        if kept_total == 0 and tried <= 2:
            print("[INFO] generating first pair...", flush=True)
            print(f"[INFO] sample_id={s.get('id')} type={s.get('type')} evidence_chunks={len(chunks)}", flush=True)

        a = generate_one(
            bundle,
            SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature_a,
            top_p=args.top_p,
            seed=base_seed,
        )
        b = generate_one(
            bundle,
            SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature_b,
            top_p=args.top_p,
            seed=base_seed + 1,
        )

        if kept_total == 0 and tried <= 2:
            print(f"[INFO] first pair generated: len(a)={len(a)} len(b)={len(b)}", flush=True)

        sa, fa = score_candidate(s["type"], a, evidence_map, evidence_text)
        sb, fb = score_candidate(s["type"], b, evidence_map, evidence_text)

        if abs(sa - sb) < args.min_margin:
            continue

        if sa > sb:
            chosen, rejected = a, b
            chosen_score, rejected_score = sa, sb
            chosen_feat, rejected_feat = fa, fb
        else:
            chosen, rejected = b, a
            chosen_score, rejected_score = sb, sa
            chosen_feat, rejected_feat = fb, fa

        row = {
            "prompt": build_full_prompt(SYSTEM_PROMPT, user_prompt),
            "chosen": chosen,
            "rejected": rejected,
            "meta": {
                "id": s.get("id"),
                "type": s.get("type"),
                "evidence_chunk_ids": ev_ids,
                "score_chosen": chosen_score,
                "score_rejected": rejected_score,
                "margin": abs(sa - sb),
                "feat_chosen": chosen_feat,
                "feat_rejected": rejected_feat,
            },
        }

        # Decide split
        if kept_eval < target_eval:
            split = "eval"
            kept_eval += 1
        else:
            split = "train"
            kept_train += 1
        kept_total = kept_eval + kept_train

        if args.incremental:
            buffer.append((row, split))
            if len(buffer) >= max(1, args.flush_every):
                flush_buffer()
        else:
            buffer.append((row, split))

    # Final flush
    if args.incremental:
        flush_buffer()
    else:
        # materialize buffers into lists
        eval_rows = [r for r, sp in buffer if sp == "eval"]
        train_rows = [r for r, sp in buffer if sp == "train"]
        write_jsonl(out_train, train_rows)
        write_jsonl(out_eval, eval_rows)

    # Report (in incremental mode we count lines from disk)
    final_eval = count_jsonl_lines(out_eval)
    final_train = count_jsonl_lines(out_train)
    kept_pairs = final_eval + final_train

    report = {
        "requested_pairs": args.num_pairs,
        "kept_pairs": kept_pairs,
        "tried": tried,
        "keep_rate": (kept_pairs / tried) if tried else 0,
        "eval_size": final_eval,
        "train_size": final_train,
        "min_margin": args.min_margin,
        "model_name": args.model_name,
        "adapter_path": str(adapter) if adapter else None,
        "temps": {"a": args.temperature_a, "b": args.temperature_b, "top_p": args.top_p},
        "max_new_tokens": args.max_new_tokens,
        "incremental": args.incremental,
        "resume": args.resume,
        "flush_every": args.flush_every,
        "progress_every": args.progress_every,
    }

    DPO_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] wrote DPO datasets:")
    print(f"  train: {out_train} ({final_train})")
    print(f"  eval : {out_eval} ({final_eval})")
    print("[OK] quality report:")
    print(REPORT_OUT)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
