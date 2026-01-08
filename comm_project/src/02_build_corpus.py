# -*- coding: utf-8 -*-
"""comm_project/src/02_build_corpus.py

Day2 Step 3: Build evidence corpus (chunk-level) from parsed paragraphs.

Input:
- comm_project/data/corpus/parsed_paragraphs.jsonl

Output:
- comm_project/data/corpus/evidence_corpus.jsonl

Chunking rules (aligned to docs/comm_llm_plan.md, adapted to token-free length):
- Aggregate by (doc_id, section) preserving order.
- Split within a section into chunks by length (char-length proxy):
  - target: 800–1500 chars (approx proxy for 600–900 tokens)
  - min: 800 chars (too-short chunks must be merged or dropped)
  - hard max: 2200 chars (must not be exceeded in final output)
- chunk_id format:
  {docShort}-{section}-{index:04d}

Hard guarantees added in this version:
- Final chunks will NOT exceed HARD_MAX (post-merge re-splitting).
- Extremely short chunks are eliminated:
  - If a leftover chunk is < MIN_FINAL and cannot be merged without exceeding HARD_MAX,
    it will be appended if possible, otherwise dropped (rare edge case).

Notes:
- This script uses char-length as an approximate proxy for token length.
  You can refine later with a tokenizer if needed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
IN_PATH = CORPUS_DIR / "parsed_paragraphs.jsonl"
OUT_PATH = CORPUS_DIR / "evidence_corpus.jsonl"


def safe_doc_short(doc_id: str) -> str:
    # Keep alnum only, uppercase for stability
    s = re.sub(r"[^0-9A-Za-z]+", "", doc_id)
    return s.upper()[:24] if s else "DOC"


def normalize_section(section):
    if not section:
        return "UNKNOWN"
    return str(section).strip()


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


def hard_split_text(text: str, hard_max: int) -> list[str]:
    """Split a long text into pieces each <= hard_max.

    Prefer splitting on newlines; fallback to hard cut.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= hard_max:
        return [text]

    parts = []
    cur = []
    cur_len = 0

    for seg in text.split("\n"):
        seg = seg.strip()
        if not seg:
            continue
        add_len = len(seg) + (1 if cur else 0)
        if cur_len + add_len > hard_max and cur:
            parts.append("\n".join(cur).strip())
            cur = [seg]
            cur_len = len(seg)
        else:
            cur.append(seg)
            cur_len += add_len

    if cur:
        parts.append("\n".join(cur).strip())

    # If any part still exceeds hard_max (e.g., a single very long line), hard cut
    final_parts = []
    for p in parts:
        if len(p) <= hard_max:
            final_parts.append(p)
        else:
            start = 0
            while start < len(p):
                final_parts.append(p[start : start + hard_max])
                start += hard_max

    return final_parts


def chunk_section_texts(texts, *, target_min: int, target_max: int, hard_max: int, min_final: int) -> list[str]:
    """Chunk by char length using paragraph boundaries, with final guarantees.

    Pipeline:
    1) Initial chunking (paragraph boundary) targeting [target_min, target_max].
    2) Merge too-small chunks (try to keep <= hard_max).
    3) Post-merge hard-split to guarantee <= hard_max.
    4) Final elimination of extremely short leftovers (< min_final).
    """

    # 1) initial chunks
    chunks = []
    cur = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        chunks.append("\n".join(cur).strip())
        cur = []
        cur_len = 0

    for t in texts:
        t = str(t).strip()
        if not t:
            continue

        # If one paragraph itself is huge, split it immediately
        if len(t) > hard_max:
            flush()
            chunks.extend(hard_split_text(t, hard_max))
            continue

        add_len = len(t) + (1 if cur else 0)
        if cur and (cur_len + add_len > target_max) and (cur_len >= target_min):
            flush()

        cur.append(t)
        cur_len += add_len

        if cur_len >= target_max:
            flush()

    flush()

    # 2) merge too-small chunks, but never exceed hard_max
    merged = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if not merged:
            merged.append(c)
            continue

        if len(merged[-1]) < target_min:
            candidate = (merged[-1] + "\n" + c).strip()
            if len(candidate) <= hard_max:
                merged[-1] = candidate
            else:
                # can't merge without breaking hard_max; keep as separate
                merged.append(c)
        else:
            merged.append(c)

    # 3) post-merge hard split to guarantee <= hard_max
    post = []
    for m in merged:
        post.extend(hard_split_text(m, hard_max))

    # 4) eliminate extremely short leftovers
    final = []
    for c in post:
        c = c.strip()
        if not c:
            continue
        if len(c) < min_final:
            # Try merge into previous if possible
            if final:
                candidate = (final[-1] + "\n" + c).strip()
                if len(candidate) <= hard_max:
                    final[-1] = candidate
                    continue
            # Try merge into next is not possible here (single pass). Drop it.
            continue
        final.append(c)

    return final


def main() -> int:
    if not IN_PATH.exists():
        raise RuntimeError(f"Missing input: {IN_PATH}. Run 01_parse_docx.py first.")

    # Group paragraph texts by (doc_id, section)
    grouped = {}
    order_keys = []

    for row in read_jsonl(IN_PATH):
        doc_id = row.get("doc_id")
        section = normalize_section(row.get("section"))
        title = row.get("title")
        text = row.get("text")

        if not doc_id or not isinstance(text, str) or not text.strip():
            continue

        key = (doc_id, section)
        if key not in grouped:
            grouped[key] = {
                "doc_id": doc_id,
                "section": section,
                "title": title,
                "texts": [],
            }
            order_keys.append(key)

        grouped[key]["texts"].append(text.strip())

    # Chunking parameters (char-based proxy)
    TARGET_MIN = 800
    TARGET_MAX = 1500
    HARD_MAX = 2200
    MIN_FINAL = 120  # eliminate tiny fragments (e.g., 5 chars)

    out_rows = []

    for doc_id, section in order_keys:
        item = grouped[(doc_id, section)]
        title = item.get("title")
        texts = item["texts"]

        section_chunks = chunk_section_texts(
            texts,
            target_min=TARGET_MIN,
            target_max=TARGET_MAX,
            hard_max=HARD_MAX,
            min_final=MIN_FINAL,
        )

        doc_short = safe_doc_short(doc_id)
        section_safe = re.sub(r"[^0-9A-Za-z.]+", "", section) or "UNKNOWN"

        for idx, ch_text in enumerate(section_chunks, start=1):
            chunk_id = f"{doc_short}-{section_safe}-{idx:04d}"
            out_rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "section": section,
                    "title": title,
                    "text": ch_text,
                    "keywords": [],
                }
            )

    write_jsonl(OUT_PATH, out_rows)

    lengths = [len(r["text"]) for r in out_rows]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"[OK] wrote evidence corpus: {OUT_PATH}")
    print(f"  sections={len(order_keys)} chunks={len(out_rows)} avg_chars={avg_len:.1f}")
    if lengths:
        print(f"  min_chars={min(lengths)} max_chars={max(lengths)}")
        # Extra health checks
        tiny = sum(1 for L in lengths if L < MIN_FINAL)
        over = sum(1 for L in lengths if L > HARD_MAX)
        print(f"  tiny(<{MIN_FINAL})={tiny} over_hard_max(>{HARD_MAX})={over}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
