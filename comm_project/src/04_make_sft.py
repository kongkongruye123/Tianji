# -*- coding: utf-8 -*-
"""comm_project/src/04_make_sft.py

Day3–Day4 Step 5: Build SFT datasets (train/eval) from evidence corpus.

Input:
- comm_project/data/corpus/evidence_corpus.jsonl

Outputs:
- comm_project/data/sft/train.jsonl
- comm_project/data/sft/eval.jsonl
- comm_project/data/sft/quality_report.json  (quality metrics)

Training sample format (aligned to docs/comm_llm_plan.md):
{
  "conversation": [
    {
      "system": "{SYSTEM_PROMPT}",
      "input": "问题: ...\n证据:\n[chunk_id=... doc_id=... section=...]\n...",
      "output": "{JSON}" 
    }
  ]
}

We generate 4 categories per plan:
- definition-like
- condition/shall-like
- procedure/steps-like
- refusal (mismatched evidence)

Quality checks:
- json_parse_rate
- schema_pass_rate
- quote_exact_match_rate (quote must be substring of evidence text)
- refusal_ratio

Key fix in this version:
- Quotes are ALWAYS extracted as exact substrings from the evidence chunk text.
  (No sentence re-joining with different whitespace/newlines.)
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Make imports work whether running as a module or as a script from repo root
THIS_DIR = Path(__file__).resolve().parent
# add comm_project/src to sys.path so we can import utils.prompts
sys.path.insert(0, str(THIS_DIR))

from utils.prompts import SYSTEM_PROMPT, format_user_prompt, validate_json_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "evidence_corpus.jsonl"
SFT_DIR = PROJECT_ROOT / "data" / "sft"
TRAIN_PATH = SFT_DIR / "train.jsonl"
EVAL_PATH = SFT_DIR / "eval.jsonl"
REPORT_PATH = SFT_DIR / "quality_report.json"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_quote_exact(text: str, rng: random.Random, min_len: int, max_len: int):
    """Pick an exact substring from text.

    This guarantees: quote in text.
    We avoid reformatting whitespace/newlines.
    """
    text = text or ""
    text = text.strip()
    if len(text) < min_len:
        return None

    # Prefer selecting within a single line or a few consecutive lines
    lines = text.split("\n")
    lines = [ln for ln in lines if ln.strip()]
    if lines:
        # pick a start line and take 1-3 lines
        for _ in range(10):
            i = rng.randrange(0, len(lines))
            take = rng.choice([1, 2, 3])
            block = "\n".join(lines[i : i + take]).strip()
            if min_len <= len(block) <= max_len and block in text:
                return block

    # fallback: random window
    for _ in range(20):
        L = rng.randrange(min_len, min(max_len, len(text)) + 1)
        start = rng.randrange(0, len(text) - L + 1)
        quote = text[start : start + L]
        quote = quote.strip()
        if min_len <= len(quote) <= max_len and quote in text:
            return quote

    return None


def make_output_answerable(chunk: dict, quote: str, answer: str, confidence: str = "mid") -> dict:
    return {
        "answer": answer,
        "citations": [
            {
                "doc_id": chunk.get("doc_id", ""),
                "section": chunk.get("section", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "quote": quote,
            }
        ],
        "confidence": confidence,
        "cannot_answer_reason": None,
    }


def make_output_refusal(reason: str, confidence: str = "low") -> dict:
    return {
        "answer": "",
        "citations": [],
        "confidence": confidence,
        "cannot_answer_reason": reason,
    }


def build_evidence_text(chunks):
    lines = []
    for ch in chunks:
        lines.append(
            f"[chunk_id={ch['chunk_id']} doc_id={ch['doc_id']} section={ch['section']}]\n{ch['text']}"
        )
    return "\n\n".join(lines)


def make_samples_from_chunk(chunk: dict, rng: random.Random):
    """Generate 0-3 answerable samples from one chunk."""
    samples = []
    text = chunk["text"]

    # Definition-like
    if re.search(r"\b(is|refers to|defined as)\b", text, re.IGNORECASE):
        quote = pick_quote_exact(text, rng, min_len=60, max_len=240)
        if quote:
            q = f"What does the evidence define or state about: {chunk.get('title') or 'this concept'}?"
            ans = quote
            out = make_output_answerable(chunk, quote, ans, confidence="high")
            samples.append((q, [chunk], out, "definition"))

    # Condition/shall-like
    if re.search(r"\bshall\b|\bwhen\b|\bif\b", text, re.IGNORECASE):
        quote = pick_quote_exact(text, rng, min_len=60, max_len=240)
        if quote:
            q = "Under what condition does the specification require the described behavior?"
            ans = quote
            out = make_output_answerable(chunk, quote, ans, confidence="mid")
            samples.append((q, [chunk], out, "condition"))

    # Procedure/steps-like: take a longer quote window, answer as bullet points of the lines
    quote = pick_quote_exact(text, rng, min_len=180, max_len=650)
    if quote:
        q = "Describe the steps or process mentioned in the evidence."
        # Bulletize by lines (keeps groundedness; avoids inventing extra steps)
        steps = [ln.strip() for ln in quote.split("\n") if ln.strip()]
        ans = "\n".join([f"- {s}" for s in steps[:10]])
        out = make_output_answerable(chunk, quote, ans, confidence="mid")
        samples.append((q, [chunk], out, "procedure"))

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--refusal_ratio", type=float, default=0.2)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if not CORPUS_PATH.exists():
        raise RuntimeError(f"Missing corpus: {CORPUS_PATH}. Run 02_build_corpus.py first.")

    corpus = list(read_jsonl(CORPUS_PATH))
    if len(corpus) < 200:
        raise RuntimeError(f"Corpus too small: {len(corpus)} chunks")

    rng.shuffle(corpus)

    total = args.train + args.eval
    target_refusals = int(round(total * args.refusal_ratio))
    target_answerables = total - target_refusals

    sft_rows = []

    # Generate answerable samples
    for ch in corpus:
        for (q, ev_chunks, out_obj, _kind) in make_samples_from_chunk(ch, rng):
            # Safety: quote must be exact substring
            quote = out_obj["citations"][0]["quote"]
            if quote not in ch["text"]:
                continue

            evidence_text = build_evidence_text(ev_chunks)
            user_prompt = format_user_prompt(q, evidence_text=evidence_text)
            out_str = json.dumps(out_obj, ensure_ascii=False)

            sft_rows.append(
                {
                    "conversation": [
                        {
                            "system": SYSTEM_PROMPT.strip(),
                            "input": user_prompt.strip(),
                            "output": out_str,
                        }
                    ]
                }
            )

            if len(sft_rows) >= target_answerables:
                break
        if len(sft_rows) >= target_answerables:
            break

    # Generate refusal samples by mismatching evidence
    refusal_rows = []
    for _ in range(target_refusals * 3):  # oversample then cut
        src = rng.choice(corpus)
        quote = pick_quote_exact(src["text"], rng, min_len=40, max_len=180)
        if not quote:
            continue
        q = f"According to the evidence, what does the specification state about: {quote[:120]}...?"
        evidence = rng.choice(corpus)
        if evidence["chunk_id"] == src["chunk_id"]:
            continue

        evidence_text = build_evidence_text([evidence])
        user_prompt = format_user_prompt(q, evidence_text=evidence_text)
        out_obj = make_output_refusal("The provided evidence does not contain the information needed to answer the question.")
        out_str = json.dumps(out_obj, ensure_ascii=False)

        refusal_rows.append(
            {
                "conversation": [
                    {
                        "system": SYSTEM_PROMPT.strip(),
                        "input": user_prompt.strip(),
                        "output": out_str,
                    }
                ]
            }
        )
        if len(refusal_rows) >= target_refusals:
            break

    sft_rows.extend(refusal_rows)
    rng.shuffle(sft_rows)

    # Split train/eval
    train_rows = sft_rows[: args.train]
    eval_rows = sft_rows[args.train : args.train + args.eval]

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    # Quality report
    total_outputs = len(sft_rows)
    json_ok = 0
    schema_ok = 0
    quote_ok = 0
    refusal = 0

    evidence_map = {c["chunk_id"]: c for c in corpus}

    for row in sft_rows:
        out_str = row["conversation"][0]["output"]
        try:
            obj = json.loads(out_str)
            json_ok += 1
        except Exception:
            continue

        ok, _msg = validate_json_output(obj)
        if ok:
            schema_ok += 1

        if obj.get("cannot_answer_reason") is not None:
            refusal += 1

        cits = obj.get("citations") or []
        if obj.get("cannot_answer_reason") is None and cits:
            cit = cits[0]
            chunk_id = cit.get("chunk_id")
            quote = cit.get("quote")
            ev = evidence_map.get(chunk_id)
            if ev and isinstance(quote, str) and quote in ev.get("text", ""):
                quote_ok += 1

    report = {
        "total": total_outputs,
        "json_parse_rate": json_ok / total_outputs if total_outputs else 0,
        "schema_pass_rate": schema_ok / total_outputs if total_outputs else 0,
        "quote_exact_match_rate": quote_ok / max(1, (total_outputs - refusal)),
        "refusal_ratio": refusal / total_outputs if total_outputs else 0,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] wrote SFT datasets:")
    print(f"  train: {TRAIN_PATH} ({len(train_rows)})")
    print(f"  eval : {EVAL_PATH} ({len(eval_rows)})")
    print("[OK] quality report:")
    print(REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
