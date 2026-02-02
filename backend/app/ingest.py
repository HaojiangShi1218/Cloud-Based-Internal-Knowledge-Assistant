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
import hashlib

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

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

def chunk_text(text: str, chunk_size: int = 3200, overlap: int = 300) -> List[str]:
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
                doc_id = file_sha256(p)
                pages = read_pdf_pages(p)

                MAX_PAGES_PER_PDF = int(os.getenv("MAX_PAGES_PER_PDF", "0"))
                if MAX_PAGES_PER_PDF > 0:
                    pages = pages[:MAX_PAGES_PER_PDF]

                logger.info(f"Ingesting {p.name} doc_id={doc_id[:8]} pages={len(pages)}")

                # Build sliding windows: page i + page i+1
                for i in range(len(pages)):
                    page_num, page_text = pages[i]
                    page_text = clean_pdf_text(page_text)

                    next_text = ""
                    page_end = page_num

                    if i + 1 < len(pages):
                        next_page_num, next_page_text = pages[i + 1]
                        next_page_text = clean_pdf_text(next_page_text)
                        next_text = "\n\n" + next_page_text
                        page_end = next_page_num

                    combined = (page_text + next_text).strip()
                    if not combined:
                        continue

                    docs.append(
                        (
                            p.name,
                            combined,
                            {
                                "page_num": page_num,      # starting page
                                "page_end": page_end,      # ending page (same as start if last page)
                                "doc_id": doc_id,
                            },
                        )
                    )

            elif ext in [".txt", ".md"]:
                doc_id = file_sha256(p)
                logger.info(f"Ingesting {p.name} doc_id={doc_id[:8]}")

                text = p.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    docs.append((p.name, text, {"page_num": None, "page_end": None, "doc_id": doc_id}))

            else:
                continue  # ignore other files safely

        except Exception as e:
            logger.warning(f"Failed reading {p.name}: {e}")

    return docs


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
    logger.info(f"Loaded {len(docs)} document pages/entries")

    if not docs:
        logger.error("No documents found. Put PDFs/txt/md into docs/ and retry.")
        return

    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)

    index_path = Path(settings.FAISS_INDEX_PATH)
    meta_path = Path(settings.META_PATH)

    index = None
    meta: List[Dict[str, Any]] = []
    seen = set()
    next_id = 0

    # ---- Resume ----
    if index_path.exists() and meta_path.exists():
        logger.info("Found existing index/meta — resuming (dedupe enabled).")
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        next_id = len(meta)
        for m in meta:
            doc_id = m.get("doc_id")
            if not doc_id:
                continue  # old entries; can't safely dedupe
            seen.add((doc_id, m.get("page_num"), m.get("page_end"), m.get("chunk_index")))

        logger.info(f"Resumed: index.ntotal={index.ntotal}, meta={len(meta)}, seen={len(seen)}")

        if index.ntotal != len(meta):
            logger.warning(
                f"Mismatch: index.ntotal={index.ntotal} meta={len(meta)}. "
                "Likely a prior crash mid-write. Consider rebuilding once if issues appear."
            )

    # ---- Incremental batching ----
    BATCH_SIZE = 128
    SAVE_EVERY_CHUNKS = 2000

    chunk_buf: List[str] = []
    meta_buf: List[Dict[str, Any]] = []

    def checkpoint():
        if index is None:
            return
        faiss.write_index(index, settings.FAISS_INDEX_PATH)
        with open(settings.META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: chunks={len(meta)}")

    def flush():
        nonlocal index, meta, chunk_buf, meta_buf
        if not chunk_buf:
            return

        emb = embed_texts(chunk_buf)
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        faiss.normalize_L2(emb)
        dim = emb.shape[1]

        if index is None:
            logger.info(f"Creating FAISS index (dim={dim})")
            index = faiss.IndexFlatIP(dim)

        index.add(emb)
        meta.extend(meta_buf)

        chunk_buf.clear()
        meta_buf.clear()

        if len(meta) % SAVE_EVERY_CHUNKS < BATCH_SIZE:
            checkpoint()

    added = 0
    skipped = 0

    for (source, text, extra) in docs:
        doc_id = extra.get("doc_id")
        page_num = extra.get("page_num")
        page_end = extra.get("page_end", page_num)

        chunks = chunk_text(text)
        for chunk_index, c in enumerate(chunks):
            key = (doc_id, page_num, page_end, chunk_index)
            if key in seen:
                skipped += 1
                continue

            seen.add(key)

            chunk_buf.append(c)
            meta_buf.append({
                "id": next_id,
                "doc_id": doc_id,
                "source": source,
                "page_num": page_num,
                "page_end": page_end,
                "chunk_index": chunk_index,
                "text": c,
            })
            next_id += 1
            added += 1

            if len(chunk_buf) >= BATCH_SIZE:
                flush()

    flush()
    checkpoint()

    logger.success(f"Done. Added={added} Skipped={skipped} Total={len(meta)}")

if __name__ == "__main__":
    main()
