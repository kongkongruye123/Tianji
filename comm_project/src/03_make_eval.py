# -*- coding: utf-8 -*-
"""comm_project/src/03_make_eval.py

Day3 Step 4: Build a fixed-evidence evaluation set.

Input:
- comm_project/data/corpus/evidence_corpus.jsonl

Output:
- comm_project/data/eval/eval_set.jsonl

Design goal (from docs/comm_llm_plan.md):
- Fixed-evidence eval: each sample contains evidence_chunk_ids.
- 200–300 samples total.
- Mix answerable/unanswerable.

This script generates a *reproducible* eval set using a fixed RNG seed.
It uses simple rule-based question generation from evidence chunks.

Sample schema:
{
  "id":"E0001",
  "type":"answerable"|"unanswerable",
  "question":"...",
  "evidence_chunk_ids":["..."],
  "must_quotes":["..."],
  "notes":"..."
}

Notes:
- must_quotes are exact substrings from the evidence chunk text.
- unanswerable samples are built by mismatching evidence with a question derived from a different chunk.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "evidence_corpus.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "eval"
OUT_PATH = OUT_DIR / "eval_set.jsonl"


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


def split_sentences(text: str):
    # Very simple sentence splitter for English-heavy 3GPP specs.
    # Keep punctuation.
    parts = re.split(r"(?<=[\.\?\!;:])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def pick_quote_from_chunk(text: str, rng: random.Random) -> str | None:
    sents = split_sentences(text)
    # Prefer medium-length sentences for stable quoting
    candidates = [s for s in sents if 60 <= len(s) <= 240]
    if not candidates:
        candidates = [s for s in sents if 30 <= len(s) <= 300]
    if not candidates:
        return None
    return rng.choice(candidates)


def make_answerable_question(quote: str, chunk: dict, rng: random.Random) -> str:
    # Simple templates
    templates = [
        "According to the provided evidence, what does the specification state about: {topic}?",
        "Based on the evidence, explain the requirement described: {topic}.",
        "What is described in the evidence regarding: {topic}?",
        "Summarize what the evidence says about: {topic}.",
    ]

    # topic: take a fragment of quote
    topic = quote
    # shorten topic for question
    if len(topic) > 120:
        topic = topic[:120].rstrip() + "..."

    # Occasionally ask about section topic
    if rng.random() < 0.25 and chunk.get("title"):
        topic = chunk["title"]

    return rng.choice(templates).format(topic=topic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--total", type=int, default=240, help="Total eval samples (200-300 recommended)")
    ap.add_argument("--unanswerable", type=float, default=0.33, help="Fraction of unanswerable")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if not CORPUS_PATH.exists():
        raise RuntimeError(f"Missing corpus: {CORPUS_PATH}. Run 02_build_corpus.py first.")

    corpus = list(read_jsonl(CORPUS_PATH))
    if len(corpus) < 200:
        raise RuntimeError(f"Corpus too small: {len(corpus)} chunks")

    total = args.total
    unans_n = int(round(total * args.unanswerable))
    ans_n = total - unans_n

    # Sample chunks for answerable and for question sources
    rng.shuffle(corpus)
    answerable_chunks = corpus[: ans_n]
    question_source_chunks = corpus[ans_n : ans_n + unans_n]

    rows = []
    idx = 1

    # Answerable
    for ch in answerable_chunks:
        quote = pick_quote_from_chunk(ch["text"], rng)
        if not quote:
            continue
        q = make_answerable_question(quote, ch, rng)
        rows.append(
            {
                "id": f"E{idx:04d}",
                "type": "answerable",
                "question": q,
                "evidence_chunk_ids": [ch["chunk_id"]],
                "must_quotes": [quote],
                "notes": f"doc={ch.get('doc_id')} section={ch.get('section')} title={ch.get('title')}",
            }
        )
        idx += 1
        if idx > ans_n:
            break

    # Unanswerable (mismatch evidence)
    for src in question_source_chunks:
        quote = pick_quote_from_chunk(src["text"], rng)
        if not quote:
            continue
        q = make_answerable_question(quote, src, rng)

        # Pick a different chunk as evidence (ideally different doc/section)
        evidence = rng.choice(corpus)
        tries = 0
        while tries < 10 and evidence["chunk_id"] == src["chunk_id"]:
            evidence = rng.choice(corpus)
            tries += 1

        rows.append(
            {
                "id": f"E{idx:04d}",
                "type": "unanswerable",
                "question": q,
                "evidence_chunk_ids": [evidence["chunk_id"]],
                "must_quotes": [quote],
                "notes": f"question_from={src.get('doc_id')}:{src.get('section')} but evidence={evidence.get('doc_id')}:{evidence.get('section')}",
            }
        )
        idx += 1
        if len(rows) >= total:
            break

    # If we couldn't hit total due to quote failures, top up randomly
    while len(rows) < total:
        ch = rng.choice(corpus)
        quote = pick_quote_from_chunk(ch["text"], rng)
        if not quote:
            continue
        q = make_answerable_question(quote, ch, rng)
        rows.append(
            {
                "id": f"E{len(rows)+1:04d}",
                "type": "answerable",
                "question": q,
                "evidence_chunk_ids": [ch["chunk_id"]],
                "must_quotes": [quote],
                "notes": "topped_up",
            }
        )

    out_path = Path(args.out)
    write_jsonl(out_path, rows)

    ans = sum(1 for r in rows if r["type"] == "answerable")
    unans = sum(1 for r in rows if r["type"] == "unanswerable")
    print(f"[OK] wrote eval set: {out_path}")
    print(f"  total={len(rows)} answerable={ans} unanswerable={unans} seed={args.seed}")


if __name__ == "__main__":
    main()

