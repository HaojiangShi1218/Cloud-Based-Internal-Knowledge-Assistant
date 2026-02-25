import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.embeddings import embed_texts

from pypdf import PdfReader
from loguru import logger
from opensearchpy import helpers
# from openai import OpenAI
from app.config import settings
from app.vectorstores.opensearch_store import get_os_client
import re
import hashlib
from collections import defaultdict

CHUNK_SIZE = 2600
CHUNK_OVERLAP = 600

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

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
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

    BATCH_SIZE = 128
    client = get_os_client()
    index_name = settings.OPENSEARCH_INDEX
    chunk_buf: List[str] = []
    docs_buf: List[Dict[str, Any]] = []
    total = 0

    def flush():
        nonlocal total
        if not chunk_buf:
            return
        emb = embed_texts(chunk_buf).tolist()
        now = datetime.now(timezone.utc).isoformat()
        actions = []
        for d, vec in zip(docs_buf, emb):
            doc_id = d["doc_id"]
            page = d["page"]
            chunk_id = d["chunk_id"]
            op_id = f"{doc_id}:{page}:{chunk_id}"
            source_doc = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_chunk_seq": d["doc_chunk_seq"],
                "page": page,
                "page_end": d.get("page_end", page),
                "source": d["source"],
                "title": d["title"],
                "text": d["text"],
                "embedding": vec,
                "created_at": now,
            }
            actions.append(
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": op_id,
                    "_source": source_doc,
                }
            )

        helpers.bulk(client, actions, request_timeout=120, refresh=False)
        total += len(actions)
        logger.info(f"Indexed {total} chunks to OpenSearch")
        chunk_buf.clear()
        docs_buf.clear()

    doc_seq_counters: Dict[str, int] = defaultdict(int)

    for (source, text, extra) in docs:
        doc_id = extra.get("doc_id")
        page_num = extra.get("page_num")
        page_end = extra.get("page_end", page_num)
        chunks = chunk_text(text)
        for chunk_index, c in enumerate(chunks):
            chunk_buf.append(c)
            docs_buf.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "title": source,
                    "page": page_num if page_num is not None else 0,
                    "page_end": page_end if page_end is not None else (page_num if page_num is not None else 0),
                    "chunk_id": chunk_index,
                    "doc_chunk_seq": doc_seq_counters.get(doc_id, 0),
                    "text": c,
                }
            )
            doc_seq_counters[doc_id] = doc_seq_counters.get(doc_id, 0) + 1

            if len(chunk_buf) >= BATCH_SIZE:
                flush()

    flush()
    client.indices.refresh(index=index_name)
    logger.success(f"Done. Indexed total={total} into {index_name}")

if __name__ == "__main__":
    main()
