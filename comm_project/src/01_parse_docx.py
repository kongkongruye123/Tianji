# -*- coding: utf-8 -*-
"""comm_project/src/01_parse_docx.py

Day2 Step 2 (DOCX variant): Parse 3GPP .docx specs into paragraph units JSONL.

Input:
- comm_project/data/raw/**/<spec>.docx  (downloaded & extracted on Day1)

Output:
- comm_project/data/corpus/parsed_paragraphs.jsonl
  One JSON object per line, roughly aligned to the plan's paragraph unit format:
    {"doc_id":..., "page": null, "section":..., "title":..., "text":...}

Cleaning rules (aligned to docs/comm_llm_plan.md, adapted for DOCX):
- Drop empty lines.
- Drop likely page headers/footers by frequency-based removal of short repeated lines.
- Drop page-number like lines: pure digits or "Page x of y".
- Skip table-derived noise: lines with many '|' or excessive spaces/alignment artifacts.
- Section detection: regex ^\d+(\.\d+)+ (e.g., 6.2.1) at line start.
- Title: best-effort (if a section heading line contains extra text after the number).

Usage:
  python comm_project/src/01_parse_docx.py

Notes:
- DOCX has no stable page concept; we set page=null.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import docx  # python-docx
except ImportError as e:
    raise RuntimeError(
        "python-docx is required. Install with: python -m pip install python-docx"
    ) from e


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # comm_project/
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "corpus"
OUT_PATH = OUT_DIR / "parsed_paragraphs.jsonl"

# Regex to capture section headings like: "5.4.3.2", "Annex L", "K.2.2.5"
SECTION_RE = re.compile(
    r"^(?P<section>"
    r"(?:Annex\s+[A-Z])"
    r"|(?:[A-Z](?:\.\d+)+)"
    r"|(?:\d+(?:\.\d+)*)"
    r")\b\s*(?P<title>.*)$"
)
PAGE_RE = re.compile(r"^Page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)


def is_page_number_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if s.isdigit():
        return True
    if PAGE_RE.match(s) is not None:
        return True
    return False


def is_table_noise_line(s: str) -> bool:
    """Heuristic: drop lines that look like table artifacts."""
    if not s:
        return True
    # Many pipes or box drawing
    if s.count("|") >= 3:
        return True
    if any(ch in s for ch in ("┌", "┐", "└", "┘", "─", "│")):
        return True
    # Excessive runs of spaces (alignment)
    if re.search(r"\s{6,}", s):
        return True
    return False


def is_toc_like_line(s: str) -> bool:
    """Detect TOC-like lines that end with a page number.

    Examples:
      "Annex L (normative): ... 717"
      "K.2.2.5 ... 715"
    """
    s = s.strip()
    if not s:
        return False
    # Ends with a space + 1-4 digits
    if re.search(r"\s\d{1,4}$", s):
        return True
    return False


def normalize_line(s: str) -> str:
    # Replace non-breaking spaces and normalize whitespace
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def doc_id_from_filename(path: Path) -> str:
    # Example: 23501-k00.docx -> 23501-k00
    return path.stem


def iter_docx_lines(docx_path: Path) -> list[str]:
    """Extract paragraph texts from docx (excluding empty)."""
    d = docx.Document(str(docx_path))
    lines = []
    for p in d.paragraphs:
        t = normalize_line(p.text)
        if t:
            lines.append(t)
    return lines


def detect_repeated_short_lines(all_lines: list[str], max_len: int = 80, min_count: int = 20) -> set[str]:
    """Frequency-based header/footer removal.

    For DOCX we approximate by removing short lines that repeat many times.
    Thresholds:
    - max_len: only consider short lines
    - min_count: appear at least this many times across the document
    """
    cand = [ln for ln in all_lines if 1 <= len(ln) <= max_len and not is_page_number_line(ln)]
    c = Counter(cand)
    return {k for k, v in c.items() if v >= min_count}


def parse_paragraph_units(doc_id: str, lines: list[str]) -> list[dict]:
    """Convert cleaned lines into paragraph units with section tracking."""
    units = []
    current_section = None
    current_title = None

    for ln in lines:
        m = SECTION_RE.match(ln)
        if m:
            current_section = m.group("section")
            title = normalize_line(m.group("title"))
            current_title = title if title else None
            # Do not emit heading itself as content; next lines carry content.
            continue

        units.append(
            {
                "doc_id": doc_id,
                "page": None,
                "section": current_section,
                "title": current_title,
                "text": ln,
            }
        )

    return units


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find extracted docx files under data/raw/**
    docx_files = sorted(RAW_DIR.rglob("*.docx"))
    if not docx_files:
        raise RuntimeError(
            f"No .docx found under {RAW_DIR}. Run 00_download.py first and ensure archives were extracted."
        )

    all_units = []

    for fp in docx_files:
        doc_id = doc_id_from_filename(fp)
        print(f"[PARSE] {fp} (doc_id={doc_id})")

        lines = iter_docx_lines(fp)
        if not lines:
            print(f"  [WARN] empty docx: {fp}")
            continue

        repeated = detect_repeated_short_lines(lines)

        cleaned = []
        for ln in lines:
            if ln in repeated:
                continue
            if is_page_number_line(ln):
                continue
            if is_table_noise_line(ln):
                continue
            # Drop TOC-like lines (title + trailing page number)
            if is_toc_like_line(ln):
                continue
            cleaned.append(ln)

        units = parse_paragraph_units(doc_id=doc_id, lines=cleaned)
        all_units.extend(units)

        print(
            f"  lines={len(lines)} cleaned={len(cleaned)} repeated_removed={len(repeated)} units={len(units)}"
        )

    # Write JSONL
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for u in all_units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    print(f"\n[OK] wrote {len(all_units)} paragraph units -> {OUT_PATH}")

    # Quick sampling hint (manual check)
    sample_n = min(10, len(all_units))
    if sample_n > 0:
        print("[SAMPLE] first few lines (for sanity):")
        for u in all_units[:sample_n]:
            print(f"  - section={u['section']} title={u['title']} text={u['text'][:80]}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise

