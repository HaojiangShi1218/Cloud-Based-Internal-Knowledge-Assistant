import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.embeddings import embed_texts

import numpy as np
import faiss
faiss.omp_set_num_threads(1)
from pypdf import PdfReader
from loguru import logger
# from openai import OpenAI
from app.config import settings
import re

def read_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_num, page_text). page_num is 1-indexed."""
    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        t = (page.extract_text() or "").strip()
        if t:
            pages.append((i + 1, t))
    return pages

def clean_pdf_text(t: str) -> str:
    t = t.replace("\r", "\n")
    # join broken words like "architec-\nture" -> "architecture"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # turn single newlines into spaces, keep paragraph breaks
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # collapse excessive whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Simple character-based chunking for MVP.
    Later we can upgrade to token-based chunking.
    """
    text = text.replace("\r", "\n")
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def load_documents(docs_dir: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Returns list of (source_name, text).
    Supports: .pdf, .txt, .md
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path.resolve()}")

    docs = []
    for p in docs_path.rglob("*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        try:
            if ext == ".pdf":
                for page_num, page_text in read_pdf_pages(p):
                    page_text = clean_pdf_text(page_text)
                    docs.append((p.name, page_text, {"page_num": page_num}))
            elif ext in [".txt", ".md"]:
                docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore"), {}))
        except Exception as e:
            logger.warning(f"Failed reading {p.name}: {e}")

    return [(name, text, md) for (name, text, md) in docs if text and text.strip()]


# def embed_texts(texts: List[str]) -> np.ndarray:
#     if not settings.OPENAI_API_KEY:
#         raise RuntimeError("OPENAI_API_KEY is missing. Set it in backend/.env")

#     client = OpenAI(api_key=settings.OPENAI_API_KEY)

#     # Embedding model name may change; keep this in one place for easy swap.
#     model = "text-embedding-3-small"

#     vectors = []
#     batch_size = 64
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i : i + batch_size]
#         resp = client.embeddings.create(model=model, input=batch)
#         vectors.extend([d.embedding for d in resp.data])

#     arr = np.array(vectors, dtype=np.float32)
#     return arr


def main():
    logger.info(f"Loading documents from: {Path(settings.DOCS_DIR).resolve()}")
    docs = load_documents(settings.DOCS_DIR)
    logger.info(f"Loaded {len(docs)} documents")

    if not docs:
        logger.error("No documents found. Put PDFs/txt/md into backend/docs/ and retry.")
        return

    all_chunks: List[str] = []
    meta: List[Dict[str, Any]] = []

    for (source, text, extra) in docs:
        chunks = chunk_text(text)
        for idx, c in enumerate(chunks):
            all_chunks.append(c)
            meta.append({
                "id": len(meta),
                "source": source,
                "page_num": extra.get("page_num"),
                "chunk_index": idx,
                "text": c
            })

    logger.info(f"Total chunks: {len(all_chunks)}")

    logger.info("Generating embeddings...")
    embeddings = embed_texts(all_chunks)
    dim = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dim}")

    logger.info("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors are normalized
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, settings.FAISS_INDEX_PATH)

    with open(settings.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.success(f"Saved FAISS index to: {settings.FAISS_INDEX_PATH}")
    logger.success(f"Saved metadata to: {settings.META_PATH}")


if __name__ == "__main__":
    main()
