# -*- coding: utf-8 -*-
"""comm_project/src/04_make_sft.py

Day3–Day5 Step 5: Build SFT datasets (train/eval) from evidence corpus.

This version is tuned for Day5 "SFT v2" quality improvement, especially:
- Improve citation_exact_match_rate by increasing answerable samples with mandatory citations.
- Reduce schema drift with schema_anchor samples.

Inputs:
- comm_project/data/corpus/evidence_corpus.jsonl

Outputs:
- comm_project/data/sft/train.jsonl
- comm_project/data/sft/eval.jsonl
- comm_project/data/sft/quality_report.json

Notes:
- quote is ALWAYS exact substring of evidence chunk text.
- Outputs follow docs/comm_llm_plan.md schema 0.4.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
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
    text = (text or "").strip()
    if len(text) < min_len:
        return None

    # Prefer a single line (more stable exact match) then 2-3 lines.
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        # 1 line first
        for _ in range(12):
            i = rng.randrange(0, len(lines))
            block = lines[i].strip()
            if min_len <= len(block) <= max_len and block in text:
                return block
        # 2-3 lines
        for _ in range(12):
            i = rng.randrange(0, len(lines))
            take = rng.choice([2, 3])
            block = "\n".join(lines[i : i + take]).strip()
            if min_len <= len(block) <= max_len and block in text:
                return block

    # Fallback window
    for _ in range(30):
        L = rng.randrange(min_len, min(max_len, len(text)) + 1)
        start = rng.randrange(0, len(text) - L + 1)
        q = text[start : start + L].strip()
        if min_len <= len(q) <= max_len and q in text:
            return q

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
    blocks = []
    for ch in chunks:
        blocks.append(
            f"[chunk_id={ch['chunk_id']} doc_id={ch['doc_id']} section={ch['section']}]\n{ch['text']}"
        )
    return "\n\n".join(blocks)


def add_row(rows, question: str, evidence_chunks, out_obj: dict):
    evidence_text = build_evidence_text(evidence_chunks)
    user_prompt = format_user_prompt(question, evidence_text=evidence_text)
    out_str = json.dumps(out_obj, ensure_ascii=False)
    rows.append(
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


def make_schema_anchor_answerable(chunk: dict, rng: random.Random):
    # Strong instruction + exact single-line quote
    quote = pick_quote_exact(chunk["text"], rng, min_len=80, max_len=220)
    if not quote:
        return None
    q = (
        "You MUST output a JSON object that strictly matches the required schema. "
        "Answer using ONLY the evidence. Provide exactly 1 citation and set quote as an exact substring."
    )
    # Answer is a string; keep concise but grounded
    ans = quote
    return q, [chunk], make_output_answerable(chunk, quote, ans, confidence="high")


def make_schema_anchor_refusal(evidence_chunk: dict):
    q = (
        "You MUST output a JSON object that strictly matches the required schema. "
        "If the provided evidence is insufficient, refuse with cannot_answer_reason and empty citations."
    )
    out = make_output_refusal("The provided evidence is insufficient to answer the question.", confidence="low")
    return q, [evidence_chunk], out


def make_regular_answerables(chunk: dict, rng: random.Random):
    text = chunk["text"]
    res = []

    # definition
    if re.search(r"\b(is|refers to|defined as)\b", text, re.IGNORECASE):
        quote = pick_quote_exact(text, rng, min_len=60, max_len=240)
        if quote:
            q = f"What does the evidence define or state about: {chunk.get('title') or 'this concept'}?"
            out = make_output_answerable(chunk, quote, quote, confidence="high")
            res.append((q, [chunk], out))

    # condition/shall
    if re.search(r"\bshall\b|\bwhen\b|\bif\b", text, re.IGNORECASE):
        quote = pick_quote_exact(text, rng, min_len=60, max_len=240)
        if quote:
            q = "Under what condition does the specification require the described behavior?"
            out = make_output_answerable(chunk, quote, quote, confidence="mid")
            res.append((q, [chunk], out))

    # procedure
    quote = pick_quote_exact(text, rng, min_len=180, max_len=650)
    if quote:
        q = "Describe the steps or process mentioned in the evidence."
        steps = [ln.strip() for ln in quote.split("\n") if ln.strip()]
        ans = "\n".join([f"- {s}" for s in steps[:10]])
        out = make_output_answerable(chunk, quote, ans, confidence="mid")
        res.append((q, [chunk], out))

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--refusal_ratio", type=float, default=0.25, help="Day5: slightly increase refusals")
    ap.add_argument("--schema_anchor_ratio", type=float, default=0.35, help="Day5: increase format drilling")
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
    target_schema_anchor = int(round(target_answerables * args.schema_anchor_ratio))

    rows = []

    # 1) schema anchor answerables
    anchor_added = 0
    for ch in corpus:
        item = make_schema_anchor_answerable(ch, rng)
        if not item:
            continue
        q, ev, out = item
        qt = out["citations"][0]["quote"]
        if qt not in ch["text"]:
            continue
        add_row(rows, q, ev, out)
        anchor_added += 1
        if anchor_added >= target_schema_anchor:
            break

    # 2) regular answerables
    for ch in corpus:
        if len(rows) >= target_answerables:
            break
        for q, ev, out in make_regular_answerables(ch, rng):
            qt = out["citations"][0]["quote"]
            if qt not in ch["text"]:
                continue
            add_row(rows, q, ev, out)
            if len(rows) >= target_answerables:
                break

    # 3) refusals (mismatch evidence)
    refusal_rows = []
    for _ in range(target_refusals * 6):
        src = rng.choice(corpus)
        quote = pick_quote_exact(src["text"], rng, min_len=40, max_len=180)
        if not quote:
            continue
        q = f"According to the evidence, what does the specification state about: {quote[:120]}...?"

        evidence = rng.choice(corpus)
        if evidence["chunk_id"] == src["chunk_id"]:
            continue

        out = make_output_refusal(
            "The provided evidence does not contain the information needed to answer the question.",
            confidence="low",
        )
        add_row(refusal_rows, q, [evidence], out)
        if len(refusal_rows) >= target_refusals:
            break

    # 4) add some schema-anchor refusals (format drilling)
    anchor_refusal_rows = []
    for _ in range(max(100, target_refusals // 4)):
        evidence = rng.choice(corpus)
        q, ev, out = make_schema_anchor_refusal(evidence)
        add_row(anchor_refusal_rows, q, ev, out)

    rows.extend(refusal_rows)
    rows.extend(anchor_refusal_rows)

    rng.shuffle(rows)

    train_rows = rows[: args.train]
    eval_rows = rows[args.train : args.train + args.eval]

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    # Quality report
    total_outputs = len(train_rows) + len(eval_rows)
    json_ok = 0
    schema_ok = 0
    quote_ok = 0
    refusal = 0

    evidence_map = {c["chunk_id"]: c for c in corpus}

    for row in (train_rows + eval_rows):
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
            cid = cit.get("chunk_id")
            qt = cit.get("quote")
            ev = evidence_map.get(cid)
            if ev and isinstance(qt, str) and qt in ev.get("text", ""):
                quote_ok += 1

    report = {
        "total": total_outputs,
        "json_parse_rate": json_ok / total_outputs if total_outputs else 0,
        "schema_pass_rate": schema_ok / total_outputs if total_outputs else 0,
        "quote_exact_match_rate": quote_ok / max(1, (total_outputs - refusal)),
        "refusal_ratio": refusal / total_outputs if total_outputs else 0,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "schema_anchor_ratio": args.schema_anchor_ratio,
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
