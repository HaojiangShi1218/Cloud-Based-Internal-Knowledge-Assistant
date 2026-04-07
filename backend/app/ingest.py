import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable, Optional, Mapping
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
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

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

def _load_single_document(path: Path, source_name: Optional[str] = None) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    display_name = source_name or path.name
    result: Dict[str, Any] = {
        "path": str(path.resolve()),
        "filename": display_name,
        "doc_id": None,
        "status": "failed",
        "error": None,
        "entries_loaded": 0,
        "chunks_indexed": 0,
        "pages_processed": 0,
    }

    if not path.exists():
        result["error"] = f"File not found: {path}"
        return [], result
    if path.is_dir():
        result["error"] = f"Path is a directory, not a file: {path}"
        return [], result

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        result["status"] = "skipped"
        result["error"] = f"Unsupported file type: {ext or '<none>'}"
        return [], result

    docs: List[Tuple[str, str, Dict[str, Any]]] = []
    try:
        doc_id = file_sha256(path)
        result["doc_id"] = doc_id

        if ext == ".pdf":
            pages = read_pdf_pages(path)
            max_pages = int(os.getenv("MAX_PAGES_PER_PDF", "0"))
            if max_pages > 0:
                pages = pages[:max_pages]

            logger.info(f"Ingesting {display_name} doc_id={doc_id[:8]} pages={len(pages)}")
            result["pages_processed"] = len(pages)

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
                        display_name,
                        combined,
                        {
                            "page_num": page_num,
                            "page_end": page_end,
                            "doc_id": doc_id,
                        },
                    )
                )
        else:
            logger.info(f"Ingesting {display_name} doc_id={doc_id[:8]}")
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append((display_name, text, {"page_num": None, "page_end": None, "doc_id": doc_id}))

        result["entries_loaded"] = len(docs)
        result["status"] = "loaded" if docs else "empty"
        return docs, result
    except Exception as e:
        logger.warning(f"Failed reading {display_name}: {e}")
        result["error"] = str(e)
        return [], result


def _load_documents_from_paths(
    paths: Iterable[Path],
    source_names: Optional[Mapping[str, str]] = None,
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Dict[str, Any]]]:
    docs: List[Tuple[str, str, Dict[str, Any]]] = []
    results: List[Dict[str, Any]] = []
    for path in paths:
        source_name = None
        if source_names:
            source_name = source_names.get(str(path))
            if source_name is None:
                source_name = source_names.get(str(path.resolve()))
        loaded_docs, result = _load_single_document(path, source_name=source_name)
        docs.extend(loaded_docs)
        results.append(result)
    return docs, results


def load_documents(docs_dir: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path.resolve()}")

    paths = [p for p in docs_path.rglob("*") if p.is_file()]
    docs, _ = _load_documents_from_paths(paths)
    return docs


def _index_documents(
    docs: List[Tuple[str, str, Dict[str, Any]]],
    file_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not docs:
        return {
            "indexed_chunks": 0,
            "loaded_entries": 0,
            "files": file_results,
        }


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
    BATCH_SIZE = 128
    client = get_os_client()
    index_name = settings.OPENSEARCH_INDEX
    chunk_buf: List[str] = []
    docs_buf: List[Dict[str, Any]] = []
    total = 0
    doc_chunk_counts: Dict[str, int] = defaultdict(int)

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
            doc_chunk_counts[str(doc_id)] += 1

            if len(chunk_buf) >= BATCH_SIZE:
                flush()

    flush()
    client.indices.refresh(index=index_name)
    for result in file_results:
        doc_id = result.get("doc_id")
        result["chunks_indexed"] = doc_chunk_counts.get(str(doc_id), 0) if doc_id else 0
        if result["status"] == "loaded":
            result["status"] = "indexed" if result["chunks_indexed"] > 0 else "empty"

    logger.success(f"Done. Indexed total={total} into {index_name}")
    return {
        "indexed_chunks": total,
        "loaded_entries": len(docs),
        "files": file_results,
    }


def ingest_paths(paths: Iterable[str], source_names: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
    resolved_paths = [Path(p) for p in paths]
    docs, file_results = _load_documents_from_paths(resolved_paths, source_names=source_names)
    summary = _index_documents(docs, file_results)
    return summary["files"]


def main():
    docs_root = Path(settings.DOCS_DIR).resolve()
    logger.info(f"Loading documents from: {docs_root}")
    paths = [p for p in docs_root.rglob("*") if p.is_file()]
    docs, file_results = _load_documents_from_paths(paths)
    logger.info(f"Loaded {len(docs)} document pages/entries")

    if not docs:
        logger.error("No documents found. Put PDFs/txt/md into docs/ and retry.")
        return

    _index_documents(docs, file_results)

if __name__ == "__main__":
    main()
