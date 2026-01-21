# -*- coding: utf-8 -*-
"""
comm_project/src/utils/retrieval.py

Local retrieval (RAG) utilities for the project.

Design goals:
- No online retrieval. Only local `evidence_corpus.jsonl`.
- Persistable index (Chroma) for repeated use.
- Simple API: retrieve top-k evidence chunks for a question.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # comm_project/
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "evidence_corpus.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "chroma"
COLLECTION_NAME = "comm_evidence"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class EvidenceChunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    title: Optional[str] = None


def format_chunks_as_evidence_text(chunks: List[EvidenceChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[chunk_id={c.chunk_id} doc_id={c.doc_id} section={c.section}]\n{c.text}"
        )
    return "\n\n".join(blocks)


def _get_embeddings(model_name: str):
    # Lazy import to keep module import fast
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Silence tokenizers parallelism warning on Windows unless user explicitly wants it
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return HuggingFaceEmbeddings(model_name=model_name)


def _load_chroma(embeddings, persist_directory: Path):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def ensure_vectorstore(
    *,
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_rebuild: bool = False,
) -> Tuple[object, str]:
    """Create or load persistent Chroma vector store.

    Returns:
        (vectorstore, message)
    """
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"缺少证据库文件：{CORPUS_PATH}。请先运行："
            f"\n  python comm_project/src/01_parse_docx.py"
            f"\n  python comm_project/src/02_build_corpus.py"
        )

    embeddings = _get_embeddings(embeddings_model)

    # If index exists and not forcing rebuild, just load it.
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()) and not force_rebuild:
        vs = _load_chroma(embeddings, INDEX_DIR)
        return vs, f"已加载本地索引：{INDEX_DIR}"

    # Build index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    from langchain_core.documents import Document
    from langchain_chroma import Chroma

    docs = []
    for row in read_jsonl(CORPUS_PATH):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        # Chroma 要求所有 metadata value 必须是 str/int/float/bool，不能是 None
        def _safe(v):
            if v is None:
                return ""
            return str(v)

        meta = {
            "chunk_id": _safe(row.get("chunk_id", "")),
            "doc_id": _safe(row.get("doc_id", "")),
            "section": _safe(row.get("section", "")),
            "title": _safe(row.get("title", "")),
        }
        docs.append(Document(page_content=text, metadata=meta))

    if not docs:
        raise RuntimeError(f"证据库为空或无法解析：{CORPUS_PATH}")

    # Recreate collection
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(INDEX_DIR),
    )
    return vs, f"已构建并落盘索引：{INDEX_DIR}（docs={len(docs)}）"


def retrieve_topk(
    question: str,
    *,
    top_k: int = 5,
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[List[EvidenceChunk], str]:
    """Retrieve top-k chunks for a question."""
    if not question or not question.strip():
        return [], "问题为空，无法检索。"

    vs, msg = ensure_vectorstore(embeddings_model=embeddings_model, force_rebuild=False)
    results = vs.similarity_search_with_score(question.strip(), k=max(1, int(top_k)))

    chunks: List[EvidenceChunk] = []
    lines = [msg, f"检索结果 top_k={top_k}："]
    for i, (doc, score) in enumerate(results, 1):
        md = doc.metadata or {}
        chunk = EvidenceChunk(
            chunk_id=str(md.get("chunk_id", "")),
            doc_id=str(md.get("doc_id", "")),
            section=str(md.get("section", "")),
            title=str(md.get("title", "")) if md.get("title") else None,
            text=doc.page_content,
        )
        chunks.append(chunk)
        lines.append(
            f"{i}. chunk_id={chunk.chunk_id} doc_id={chunk.doc_id} section={chunk.section} score={score:.4f}"
        )

    return chunks, "\n".join(lines)


