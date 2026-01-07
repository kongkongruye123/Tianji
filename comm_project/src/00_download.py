# -*- coding: utf-8 -*-
"""comm_project/src/00_download.py

Day1 Step 1/3: Download a small set of public communication standards
(3GPP FTP usually provides .zip archives) and generate a reproducible manifest
with sha256 checksums.

Outputs:
- comm_project/data/raw/*.zip (local, gitignored)
- comm_project/data/raw/*.{pdf,doc,docx} (extracted, local, gitignored)
- comm_project/data/raw/manifest.json (commit this)

Design constraints (from docs/comm_llm_plan.md):
- Do NOT commit raw documents (ZIP/PDF/DOC/DOCX).
- Manifest must include: doc_id, url, filename, sha256, downloaded_at
- Script should be idempotent and fail loudly.

Notes:
- 3GPP archive file naming like "23501-k00.zip" encodes the spec version.
  Prefer selecting the newest version (largest / latest) from the directory listing.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# -------------------------
# 1) Document list (edit here)
# -------------------------
# For 3GPP official FTP, URLs typically point to a .zip archive.
# The archive usually contains one or more PDFs (and/or Word sources).
# Keep the list small for the first iteration (3–6 docs).
DOCS = [
    {
        "doc_id": "3GPP TS 23.501",
        "url": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/23501-k00.zip",
        "filename": "23501-k00.zip",
    },
    {
        "doc_id": "3GPP TS 23.502",
        "url": "https://www.3gpp.org/ftp/Specs/archive/23_series/23.502/23502-k00.zip",
        "filename": "23502-k00.zip",
    },
    {
        "doc_id": "3GPP TS 24.501",
        "url": "https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/24501-ic0.zip",
        "filename": "24501-ic0.zip",
    },
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # comm_project/
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_file(url: str, out_path: Path, timeout: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Download to temp then atomic rename (avoid half files)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "comm-llm-evidence-downloader/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", None) and resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
            with tmp_path.open("wb") as f:
                f.write(resp.read())
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Download failed: {url} -> {out_path}: {e}") from e

    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {url}")

    tmp_path.replace(out_path)


def validate_docs(docs: list[dict]) -> None:
    if not docs:
        raise RuntimeError(
            "DOCS is empty. Please edit comm_project/src/00_download.py and fill DOCS with 3–6 3GPP .zip URLs."
        )
    for i, d in enumerate(docs):
        for k in ("doc_id", "url", "filename"):
            if k not in d or not isinstance(d[k], str) or not d[k].strip():
                raise RuntimeError(f"DOCS[{i}] missing/invalid '{k}': {d}")
        if not d["filename"].lower().endswith(".zip"):
            raise RuntimeError(f"DOCS[{i}] filename must end with .zip: {d['filename']}")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    validate_docs(DOCS)

    manifest = []
    for d in DOCS:
        zip_path = RAW_DIR / d["filename"]

        # Idempotent: if exists, do not re-download by default
        if zip_path.exists() and zip_path.stat().st_size > 0:
            print(f"[SKIP] exists: {zip_path}")
        else:
            print(f"[DOWN] {d['doc_id']} -> {zip_path}")
            download_file(d["url"], zip_path)
            time.sleep(0.5)

        zip_sha256 = sha256_file(zip_path)

        # Extract all supported document types to a dedicated folder under raw/<doc_id>/
        extract_dir = RAW_DIR / d["doc_id"].replace(" ", "_")
        extract_dir.mkdir(parents=True, exist_ok=True)

        supported_exts = (".pdf", ".doc", ".docx")
        extracted_files = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # Skip directories
                if member.endswith("/"):
                    continue

                member_lower = member.lower()
                if not member_lower.endswith(supported_exts):
                    continue

                # Sanitize filename (avoid path traversal)
                file_name = Path(member).name
                target_path = extract_dir / file_name

                # Idempotent: skip if exists
                if not target_path.exists():
                    with target_path.open("wb") as f_out, zf.open(member) as f_in:
                        f_out.write(f_in.read())

                extracted_files.append(target_path)

        if not extracted_files:
            raise RuntimeError(
                f"No supported files (PDF/DOC/DOCX) found inside {zip_path}. "
                f"This may indicate the archive contains other formats only."
            )

        manifest.append(
            {
                "doc_id": d["doc_id"],
                "url": d["url"],
                "zip_filename": d["filename"],
                "zip_sha256": zip_sha256,
                "extracted_files": [
                    {"name": f.name, "sha256": sha256_file(f)} for f in extracted_files
                ],
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        print(f"  - Extracted {len(extracted_files)} files to {extract_dir}")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[OK] wrote manifest: {MANIFEST_PATH}")
    print("[OK] remember: ZIP/PDF/DOC/DOCX under comm_project/data/raw must be gitignored")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
