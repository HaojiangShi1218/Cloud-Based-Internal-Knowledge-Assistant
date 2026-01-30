import hashlib
import time
from typing import Dict, Tuple

# key -> (answer, expires_at)
_CACHE: Dict[str, Tuple[str, float]] = {}

DEFAULT_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_ITEMS = 500

def make_cache_key(question: str, hits) -> str:
    base = question.strip().lower()

    # include enough retrieval identity to invalidate when hits change
    # (source + page + chunk + small hash of text)
    parts = []
    for h in hits:
        txt = (h.get("text") or "")[:200]
        txt_hash = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
        parts.append(f"{h.get('source')}:{h.get('page_num')}:{h.get('chunk_index')}:{txt_hash}")

    raw = base + "||" + "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_cached_answer(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    answer, expires_at = item
    if time.time() > expires_at:
        _CACHE.pop(key, None)
        return None
    return answer

def set_cached_answer(key: str, answer: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    # basic cap: evict oldest-ish by clearing when full (simple MVP)
    if len(_CACHE) >= MAX_CACHE_ITEMS:
        _CACHE.clear()
    _CACHE[key] = (answer, time.time() + ttl_seconds)
