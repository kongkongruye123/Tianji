# -*- coding: utf-8 -*-
"""
comm_project/src/11_build_index.py

Engineering step: build a persistent local retrieval index (Chroma) from
`comm_project/data/corpus/evidence_corpus.jsonl`.

Usage:
  python comm_project/src/11_build_index.py
  python comm_project/src/11_build_index.py --force
"""

import argparse

from utils.retrieval import ensure_vectorstore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制重建索引")
    ap.add_argument(
        "--embeddings_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers 模型名",
    )
    args = ap.parse_args()

    _, msg = ensure_vectorstore(
        embeddings_model=args.embeddings_model, force_rebuild=bool(args.force)
    )
    print(f"[OK] {msg}")


if __name__ == "__main__":
    main()


