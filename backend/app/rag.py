import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from loguru import logger

from app.config import settings
from app.embeddings import embed_texts

MIN_SCORE = 0.25   # start here; tune later

def load_index_and_meta() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_path = Path(settings.FAISS_INDEX_PATH)
    meta_path = Path(settings.META_PATH)

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Vector store not found. Run: python -m app.ingest")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if index.ntotal != len(meta):
        raise RuntimeError(f"Index/meta mismatch: index.ntotal={index.ntotal}, len(meta)={len(meta)}")

    return index, meta


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    index, meta = load_index_and_meta()

    q = embed_texts([query]).astype(np.float32)  # (1, 384) normalized already
    # FAISS expects 2D float32
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    logger.info(f"Top FAISS scores: {scores[0].tolist()}")
    logger.info(f"Top FAISS idxs: {idxs[0].tolist()}")

    results = []
    for rank, i in enumerate(idxs[0]):
        if i == -1:
            continue

        score = float(scores[0][rank])
        # if score < MIN_SCORE:
        #     continue

        m = meta[int(i)]
        results.append({
            "rank": rank + 1,
            "score": score,
            "source": m["source"],
            "page_num": m.get("page_num"),
            "page_end": m.get("page_end", m["page_num"]),
            "chunk_index": m["chunk_index"],
            "text": m["text"],
        })
    if not results:
        logger.info(f"No results above MIN_SCORE={MIN_SCORE} for query: {query!r}")

    return results

