from typing import List
import numpy as np
from loguru import logger

from sentence_transformers import SentenceTransformer


_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading local embedding model: all-MiniLM-L6-v2")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Local embeddings: 384-dim. No API cost.
    """
    model = get_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)
