import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from loguru import logger

from app.config import settings
from app.embeddings import embed_texts
from app.llm import rewrite_queries
from app.utils.timing import span
from app.vectorstores.opensearch_store import knn_search, fetch_doc_seq_chunks

# Retrieval tuning
CANDIDATE_MULT = 12
MIN_CANDIDATES = 40
MAX_CANDIDATES = 300
BM25_ALPHA = 0.45
BM25_NORM = 8.0
BM25_K1 = 1.5
BM25_B = 0.75
PHRASE_BONUS_BIGRAM = 0.08
PHRASE_BONUS_TRIGRAM = 0.12
PHRASE_BONUS_CAP = 0.24
LIST_EXPAND_WINDOW = 4
LIST_EXPAND_MAX_EXTRA = 6
YESNO_COVERAGE_WEIGHT = 0.22
YESNO_NEGATION_BONUS = 0.08
YESNO_NEGATION_MATCH_BONUS = 0.06
YESNO_NEGATION_MISMATCH_PENALTY = 0.04
YESNO_COVERAGE_GATE = 0.22
PROXIMITY_COVERAGE_WEIGHT = 0.08
PROXIMITY_COMPACT_WEIGHT = 0.14
PROXIMITY_BONUS_CAP = 0.22
BETWEEN_BONUS = 0.08
SENT_RERANK_TOP = 8
SENT_MAX_PER_CHUNK = 6
SENT_MIN_CHARS = 40
SENT_BONUS_WEIGHT = 0.25

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "as", "what", "which", "who", "whom", "this", "that", "these", "those", "it", "its", "from",
    "by", "at", "into", "about", "how", "do", "does", "did", "can", "could", "should", "would"
}
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NEGATION_RE = re.compile(
    r"\b(not|never|no|without|cannot|can't|doesn't|isn't|aren't|won't|didn't|hasn't|haven't|hadn't)\b",
    re.IGNORECASE,
)
_YESNO_RE = re.compile(r"^\s*(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b", re.IGNORECASE)
_LIST_LEAD_RE = re.compile(
    r"(there\s+(are|is)\b[^.]{0,100}:|the\s+following\b[^.]{0,100}:)",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(r"\bbetween\b[^.]{0,120}\band\b[^.]{0,120}", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DOC_TOKEN_CACHE_MAX = 50000
_DOC_TOKEN_CACHE: Dict[Tuple[str, int], List[str]] = {}

def _tokens(s: str) -> List[str]:
    toks = [t.lower() for t in _WORD_RE.findall(s or "")]
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]

def _token_cache_key(hit: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    doc_id = hit.get("doc_id")
    seq = hit.get("doc_chunk_seq")
    if doc_id and isinstance(seq, int):
        return str(doc_id), seq
    return None

def _get_doc_tokens(hit: Dict[str, Any], text: str) -> List[str]:
    key = _token_cache_key(hit)
    if key is None:
        return _tokens(text)

    cached = _DOC_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached

    toks = _tokens(text)
    if len(_DOC_TOKEN_CACHE) >= _DOC_TOKEN_CACHE_MAX:
        _DOC_TOKEN_CACHE.clear()
    _DOC_TOKEN_CACHE[key] = toks
    return toks

def _lexical_score(query: str, text: str, qtok: Optional[List[str]] = None, tset: Optional[set] = None) -> float:
    qtok = qtok if qtok is not None else _tokens(query)
    if not qtok:
        return 0.0
    tset = tset if tset is not None else set(_tokens(text))
    if not tset:
        return 0.0
    unique_q = set(qtok)
    hits = sum(1 for t in unique_q if t in tset)
    return hits / max(1, len(unique_q))

def _compute_corpus_stats(meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    df: Counter[str] = Counter()
    total_len = 0
    for m in meta:
        toks = m.get("tokens")
        if not isinstance(toks, list):
            toks = _tokens(m.get("text", ""))
        total_len += len(toks)
        for t in set(toks):
            df[t] += 1
    n_docs = max(1, len(meta))
    avgdl = total_len / n_docs
    idf = {t: math.log((n_docs - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}
    return {"idf": idf, "avgdl": avgdl, "n_docs": n_docs}

def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms

def _bm25_score(
    query: str,
    text: str,
    stats: Dict[str, Any],
    qtok: Optional[List[str]] = None,
    toks: Optional[List[str]] = None,
) -> float:
    qtok = qtok if qtok is not None else _tokens(query)
    if not qtok:
        return 0.0
    toks = toks if toks is not None else _tokens(text)
    if not toks:
        return 0.0
    tf = Counter(toks)
    dl = len(toks)
    avgdl = stats.get("avgdl", 1.0) or 1.0
    idf = stats.get("idf", {})
    score = 0.0
    for t in set(qtok):
        f = tf.get(t, 0)
        if not f:
            continue
        term_idf = idf.get(t, 0.0)
        denom = f + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / avgdl))
        score += term_idf * (f * (BM25_K1 + 1.0)) / max(1e-6, denom)
    return score

def _bm25_norm(score: float) -> float:
    return score / (score + BM25_NORM)

def _phrase_boost(
    query: str,
    text: str,
    qtok_unique: Optional[List[str]] = None,
    text_lower: Optional[str] = None,
) -> float:
    qtok = qtok_unique if qtok_unique is not None else list(dict.fromkeys(_tokens(query)))
    if len(qtok) < 2:
        return 0.0
    t = text_lower if text_lower is not None else (text or "").lower()
    bonus = 0.0
    for i in range(len(qtok) - 1):
        if f"{qtok[i]} {qtok[i+1]}" in t:
            bonus += PHRASE_BONUS_BIGRAM
    for i in range(len(qtok) - 2):
        if f"{qtok[i]} {qtok[i+1]} {qtok[i+2]}" in t:
            bonus += PHRASE_BONUS_TRIGRAM
    return min(bonus, PHRASE_BONUS_CAP)

def _is_yesno_question(query: str) -> bool:
    return bool(_YESNO_RE.match(query or ""))

def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text or ""))

def _focus_tokens_from_qtok(qtok: List[str]) -> List[str]:
    uniq = list(dict.fromkeys(qtok))
    if not uniq:
        return []
    return sorted(uniq, key=lambda x: (-len(x), x))[:8]

def _focus_tokens(query: str) -> List[str]:
    return _focus_tokens_from_qtok(_tokens(query))

def _focus_coverage(
    query: str,
    text: str,
    focus: Optional[List[str]] = None,
    tset: Optional[set] = None,
) -> float:
    focus = focus if focus is not None else _focus_tokens(query)
    if not focus:
        return 0.0
    tset = tset if tset is not None else set(_tokens(text))
    if not tset:
        return 0.0
    hits = sum(1 for t in focus if t in tset)
    return hits / max(1, len(focus))

def _proximity_boost(
    query: str,
    text: str,
    qtok_unique: Optional[List[str]] = None,
    tks: Optional[List[str]] = None,
) -> float:
    """
    Bonus for query-token co-occurrence within a compact span of text.
    Helps favor chunks where key query terms appear together, not just anywhere.
    """
    qtok = qtok_unique if qtok_unique is not None else list(dict.fromkeys(_tokens(query)))
    if len(qtok) < 2:
        return 0.0
    tks = tks if tks is not None else [t.lower() for t in _WORD_RE.findall(text or "")]
    if not tks:
        return 0.0

    tset = set(tks)
    matched = [t for t in qtok if t in tset]
    if len(matched) < 2:
        return 0.0

    targets = set(matched)
    need = len(targets)
    have = 0
    left = 0
    counts: Dict[str, int] = {}
    min_window = None

    for right, tok in enumerate(tks):
        if tok in targets:
            counts[tok] = counts.get(tok, 0) + 1
            if counts[tok] == 1:
                have += 1

        while have == need and left <= right:
            win = right - left + 1
            if min_window is None or win < min_window:
                min_window = win
            ltok = tks[left]
            if ltok in targets:
                counts[ltok] -= 1
                if counts[ltok] == 0:
                    have -= 1
            left += 1

    if min_window is None:
        return 0.0

    coverage = len(matched) / max(1, len(qtok))
    compactness = 1.0 / (1.0 + max(0, min_window - 1) / 25.0)
    bonus = (coverage * PROXIMITY_COVERAGE_WEIGHT) + (compactness * PROXIMITY_COMPACT_WEIGHT)
    return min(bonus, PROXIMITY_BONUS_CAP)

def _between_boost(
    query: str,
    text: str,
    focus: Optional[List[str]] = None,
    query_has_between: Optional[bool] = None,
) -> float:
    """
    Generic boost for 'between ... and ...' constructions when the query
    also asks about 'between'. This is context-agnostic.
    """
    if query_has_between is None:
        query_has_between = "between" in (query or "").lower()
    if not query_has_between:
        return 0.0
    if not text:
        return 0.0
    focus = focus if focus is not None else _focus_tokens(query)
    if len(focus) < 2:
        return 0.0
    for m in _BETWEEN_RE.finditer(text):
        span = m.group(0).lower()
        hits = sum(1 for t in focus if t in span)
        if hits >= 2:
            return BETWEEN_BONUS
    return 0.0

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text.replace("\n", " "))
    sents = [p.strip() for p in parts if p and p.strip()]
    return sents

def _sentence_rerank(query: str, candidates: List[Dict[str, Any]]) -> None:
    """
    Sentence-level rerank to handle implicit questions by matching the best sentence
    inside each chunk against the query embedding.
    Embeds all candidate sentences in one batch to reduce per-call overhead.
    """
    if not candidates:
        return

    top = candidates[:SENT_RERANK_TOP]
    q = _l2_normalize(embed_texts([query]).astype(np.float32))
    all_sents: List[str] = []
    spans: List[Tuple[int, int]] = []

    for c in top:
        sents = _split_sentences(c.get("text", ""))
        sents = [s for s in sents if len(s) >= SENT_MIN_CHARS]
        if len(sents) > SENT_MAX_PER_CHUNK:
            sents = sents[:SENT_MAX_PER_CHUNK]

        start = len(all_sents)
        all_sents.extend(sents)
        end = len(all_sents)
        spans.append((start, end))

    if not all_sents:
        return

    X = _l2_normalize(embed_texts(all_sents).astype(np.float32))
    sims = (X @ q[0]).astype(np.float32)

    for c, (a, b) in zip(top, spans):
        if a == b:
            continue
        best = float(np.max(sims[a:b]))
        c["sentence_score"] = best
        c["final_score"] = float(c.get("final_score", 0.0)) + (SENT_BONUS_WEIGHT * best)

def _yesno_boost(
    query: str,
    text: str,
    focus: Optional[List[str]] = None,
    tset: Optional[set] = None,
    qneg: Optional[bool] = None,
    tneg: Optional[bool] = None,
) -> float:
    if not _is_yesno_question(query):
        return 0.0
    coverage = _focus_coverage(query, text, focus=focus, tset=tset)
    bonus = coverage * YESNO_COVERAGE_WEIGHT

    qneg = _has_negation(query) if qneg is None else qneg
    tneg = _has_negation(text) if tneg is None else tneg
    if tneg:
        bonus += YESNO_NEGATION_BONUS
    if qneg and tneg:
        bonus += YESNO_NEGATION_MATCH_BONUS
    elif qneg and not tneg:
        bonus -= YESNO_NEGATION_MISMATCH_PENALTY
    return bonus

def _is_list_question(query: str) -> bool:
    ql = (query or "").lower()
    return any(k in ql for k in [
        "what are",
        "which are",
        "list",
        "name",
        "identify",
        "enumerate",
    ])

def _is_list_lead(text: str) -> bool:
    t = text or ""
    return bool(_LIST_LEAD_RE.search(t)) or t.count("•") >= 2

def _build_doc_seq_lookup(meta: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    lut: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for m in meta:
        doc_id = m.get("doc_id")
        seq = m.get("doc_chunk_seq")
        if doc_id and isinstance(seq, int):
            lut[(doc_id, seq)] = m
    return lut

def _hit_from_meta(m: Dict[str, Any], base_score: float, base_final: float) -> Dict[str, Any]:
    return {
        "rank": 0,
        "score": base_score,
        "semantic_score": base_score,
        "lexical_score": 0.0,
        "phrase_bonus": 0.0,
        "final_score": base_final,
        "source": m["source"],
        "page_num": m.get("page_num"),
        "page_end": m.get("page_end", m["page_num"]),
        "chunk_index": m["chunk_index"],
        "doc_id": m.get("doc_id"),
        "doc_chunk_seq": m.get("doc_chunk_seq"),
        "text": m.get("text", ""),
    }

def _expand_list_context(meta: List[Dict[str, Any]], query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not hits or not _is_list_question(query):
        return hits

    lut = _build_doc_seq_lookup(meta)
    for h in hits:
        doc_id = h.get("doc_id")
        seq = h.get("doc_chunk_seq")
        if not doc_id or not isinstance(seq, int):
            continue
        lut[(doc_id, seq)] = {
            "source": h.get("source", ""),
            "page_num": h.get("page_num"),
            "page_end": h.get("page_end", h.get("page_num")),
            "chunk_index": h.get("chunk_index"),
            "doc_id": doc_id,
            "doc_chunk_seq": seq,
            "text": h.get("text", ""),
        }

    fetch_plan: Dict[str, List[int]] = {}
    for h in hits:
        doc_id = h.get("doc_id")
        seq = h.get("doc_chunk_seq")
        if not doc_id or not isinstance(seq, int):
            continue
        for off in range(1, LIST_EXPAND_WINDOW + 1):
            key = (doc_id, seq + off)
            if key in lut:
                continue
            fetch_plan.setdefault(doc_id, []).append(seq + off)

    for doc_id, seqs in fetch_plan.items():
        want = sorted(set(seqs))
        if not want:
            continue
        for m in fetch_doc_seq_chunks(doc_id, want):
            seq = m.get("doc_chunk_seq")
            if not isinstance(seq, int):
                continue
            lut[(doc_id, seq)] = {
                "source": m.get("source", ""),
                "page_num": m.get("page"),
                "page_end": m.get("page_end", m.get("page")),
                "chunk_index": m.get("chunk_id"),
                "doc_id": m.get("doc_id"),
                "doc_chunk_seq": seq,
                "text": m.get("text", ""),
            }

    seen = {
        (h.get("doc_id"), h.get("doc_chunk_seq"))
        for h in hits
        if h.get("doc_id") and isinstance(h.get("doc_chunk_seq"), int)
    }
    expanded: List[Dict[str, Any]] = []
    added = 0
    anchor_budget = 2
    used_anchors = 0

    for hit in hits:
        expanded.append(hit)
        is_anchor = _is_list_lead(hit.get("text", ""))
        if not is_anchor and used_anchors < anchor_budget and len(expanded) <= anchor_budget:
            # If no explicit list lead is found early, try expanding from top hits.
            is_anchor = True
        if not is_anchor or added >= LIST_EXPAND_MAX_EXTRA:
            continue

        doc_id = hit.get("doc_id")
        seq = hit.get("doc_chunk_seq")
        if not doc_id or not isinstance(seq, int):
            continue
        used_anchors += 1

        anchor_page = hit.get("page_num")
        for off in range(1, LIST_EXPAND_WINDOW + 1):
            if added >= LIST_EXPAND_MAX_EXTRA:
                break
            key = (doc_id, seq + off)
            if key in seen:
                continue
            m = lut.get(key)
            if not m:
                continue
            page_num = m.get("page_num")
            if isinstance(anchor_page, int) and isinstance(page_num, int) and abs(page_num - anchor_page) > 3:
                continue
            txt = m.get("text", "")
            if _lexical_score(query, txt) < 0.12 and "•" not in txt:
                continue

            expanded.append(_hit_from_meta(m, hit.get("score", 0.0), hit.get("final_score", 0.0)))
            seen.add(key)
            added += 1

    for i, r in enumerate(expanded, start=1):
        r["rank"] = i
    if added:
        logger.info(f"List-context expansion added {added} continuation chunks")
    return expanded

def retrieve(
    query: str,
    top_k: int = 5,
    expand_lists: bool = False,
    query_rewrite_enabled: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    spans: Dict[str, float] = {}
    is_yesno = _is_yesno_question(query)

    with span("query_rewrite", spans):
        queries_for_retrieval = [query]
        if query_rewrite_enabled is None:
            query_rewrite_enabled = getattr(settings, "QUERY_REWRITE_ENABLED", False)
        if query_rewrite_enabled:
            k = int(getattr(settings, "QUERY_REWRITE_K", 3))
            rewrites = rewrite_queries(query, k=k)
            if rewrites and len(rewrites) > 1:
                queries_for_retrieval = rewrites
                logger.info(f"Query rewrites: {queries_for_retrieval[1:]!r}")

    query_features = []
    for qv in queries_for_retrieval:
        qtok = _tokens(qv)
        query_features.append({
            "query": qv,
            "qtok": qtok,
            "qtok_unique": list(dict.fromkeys(qtok)),
            "focus": _focus_tokens_from_qtok(qtok),
            "has_between": "between" in (qv or "").lower(),
        })
    qneg = _has_negation(query)

    with span("embed_queries", spans):
        q_embs = embed_texts(queries_for_retrieval).astype(np.float32)
        if q_embs.ndim == 1:
            q_embs = q_embs.reshape(1, -1)

    with span("opensearch_semantic", spans):
        # Embed each query variant and fuse OpenSearch candidates by max semantic score.
        top_n = max(MIN_CANDIDATES, min(MAX_CANDIDATES, top_k * CANDIDATE_MULT))
        candidate_map: Dict[Tuple[str, int, Any], Dict[str, Any]] = {}
        for emb in q_embs:
            hits = knn_search(emb.tolist(), k=max(top_n * 2, 20))
            for h in hits:
                key = (
                    str(h.get("doc_id", "")),
                    int(h.get("chunk_id", -1) or -1),
                    h.get("page"),
                )
                semantic = float(h.get("score", 0.0))
                prev = candidate_map.get(key)
                if not prev or semantic > prev["semantic"]:
                    candidate_map[key] = {"semantic": semantic, "hit": h}

    if candidate_map:
        top_sem = sorted((v["semantic"] for v in candidate_map.values()), reverse=True)
        logger.info(f"Top OpenSearch (semantic) scores: {[round(float(s), 4) for s in top_sem[:10]]}")

    with span("bm25_lexical", spans):
        candidate_rows = []
        for v in candidate_map.values():
            hit = v["hit"]
            text = hit.get("text", "")
            doc_tokens = _get_doc_tokens(hit, text)
            candidate_rows.append({
                "hit": hit,
                "semantic": float(v["semantic"]),
                "text": text,
                "text_lower": (text or "").lower(),
                "word_tokens": [t.lower() for t in _WORD_RE.findall(text or "")],
                "doc_tokens": doc_tokens,
                "doc_token_set": set(doc_tokens),
                "has_negation": _has_negation(text),
            })

        stats = _compute_corpus_stats([{"tokens": row["doc_tokens"]} for row in candidate_rows])
        logger.info(f"Corpus stats: docs={stats['n_docs']} avgdl={round(stats['avgdl'], 1)} tokens={len(stats['idf'])}")

        candidates = []
        for row in candidate_rows:
            m = row["hit"]
            semantic = row["semantic"]
            text = row["text"]
            text_lower = row["text_lower"]
            word_tokens = row["word_tokens"]
            doc_tokens = row["doc_tokens"]
            doc_token_set = row["doc_token_set"]
            tneg = row["has_negation"]
            # Blend lexical scores across original + rewrites.
            lex_scores = [
                _lexical_score(
                    qf["query"],
                    text,
                    qtok=qf["qtok"],
                    tset=doc_token_set,
                )
                for qf in query_features
            ]
            lexical = max(lex_scores) if lex_scores else 0.0
            bm25_scores = [
                _bm25_score(
                    qf["query"],
                    text,
                    stats,
                    qtok=qf["qtok"],
                    toks=doc_tokens,
                )
                for qf in query_features
            ]
            bm25 = max(bm25_scores) if bm25_scores else 0.0
            bm25_norm = _bm25_norm(bm25)
            phrase_scores = [
                _phrase_boost(
                    qf["query"],
                    text,
                    qtok_unique=qf["qtok_unique"],
                    text_lower=text_lower,
                )
                for qf in query_features
            ]
            phrase = max(phrase_scores) if phrase_scores else 0.0
            focus_cov = _focus_coverage(
                queries_for_retrieval[0],
                text,
                focus=query_features[0]["focus"],
                tset=doc_token_set,
            )
            yesno_bonus = _yesno_boost(
                query,
                text,
                focus=query_features[0]["focus"],
                tset=doc_token_set,
                qneg=qneg,
                tneg=tneg,
            )
            proximity_scores = [
                _proximity_boost(
                    qf["query"],
                    text,
                    qtok_unique=qf["qtok_unique"],
                    tks=word_tokens,
                )
                for qf in query_features
            ]
            proximity = max(proximity_scores) if proximity_scores else 0.0
            between_scores = [
                _between_boost(
                    qf["query"],
                    text,
                    focus=qf["focus"],
                    query_has_between=qf["has_between"],
                )
                for qf in query_features
            ]
            between_bonus = max(between_scores) if between_scores else 0.0
            final = semantic + (BM25_ALPHA * bm25_norm) + phrase + yesno_bonus + proximity + between_bonus
            candidates.append({
                "rank": 0,
                "score": semantic,  # keep key name for response compatibility
                "semantic_score": semantic,
                "lexical_score": lexical,
                "bm25_score": bm25,
                "bm25_norm": bm25_norm,
                "phrase_bonus": phrase,
                "focus_coverage": focus_cov,
                "yesno_bonus": yesno_bonus,
                "proximity_bonus": proximity,
                "between_bonus": between_bonus,
                "final_score": final,
                "source": m.get("source", ""),
                "page_num": m.get("page"),
                "page_end": m.get("page_end", m.get("page")),
                "chunk_index": m.get("chunk_id"),
                "doc_id": m.get("doc_id"),
                "doc_chunk_seq": m.get("doc_chunk_seq"),
                "text": text,
            })

    with span("hybrid_merge", spans):
        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        if is_yesno:
            high_cov = [c for c in candidates if c.get("focus_coverage", 0.0) >= YESNO_COVERAGE_GATE]
            if high_cov:
                ids = {
                    (c.get("doc_id"), c.get("doc_chunk_seq"), c.get("page_num"), c.get("chunk_index"))
                    for c in high_cov
                }
                tail = [
                    c for c in candidates
                    if (c.get("doc_id"), c.get("doc_chunk_seq"), c.get("page_num"), c.get("chunk_index")) not in ids
                ]
                candidates = high_cov + tail

        out = candidates[:top_k]
        for i, r in enumerate(out, start=1):
            r["rank"] = i

        if expand_lists:
            out = _expand_list_context([], query, out)

    logger.info(f"Top hybrid final scores: {[round(r['final_score'], 4) for r in out]}")
    logger.info("RETRIEVE_SPANS {}", spans)
    return out
