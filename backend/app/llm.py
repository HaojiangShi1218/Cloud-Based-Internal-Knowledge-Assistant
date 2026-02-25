import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.cache import make_cache_key, get_cached_answer, set_cached_answer
from app.config import settings

SYSTEM_PROMPT = """You are a careful technical assistant.
Use only the provided context snippets and do not rely on outside knowledge.
Return ONLY valid JSON (no markdown).
"""

NO_ANSWER = "I don’t know based on the provided documents."
MAX_HITS_FOR_LLM = 8
MAX_CONTEXT_CHARS = 15000
MAX_CHARS_PER_HIT = 2800
MAX_CLAIMS = 8

_CITE_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_YESNO_RE = re.compile(r"^\s*(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b", re.IGNORECASE)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "as", "what", "which", "who", "whom", "this", "that", "these", "those", "it", "its", "from",
    "by", "at", "into", "about", "how", "do", "does", "did", "can", "could", "should", "would",
}

_REWRITE_CACHE: Dict[str, str] = {}


def _clip(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "..."


def _tokens(s: str) -> List[str]:
    toks = [t.lower() for t in _WORD_RE.findall(s or "")]
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]

def rewrite_queries(question: str, k: int = 3) -> List[str]:
    """
    Rewrite an implicit question into explicit search queries.
    Returns a list starting with the original question, plus up to k rewrites.
    """
    q = (question or "").strip()
    if not q:
        return [q]
    if len(_tokens(q)) < int(getattr(settings, "QUERY_REWRITE_MIN_TOKENS", 5)):
        return [q]

    key = q.lower()
    cached = _REWRITE_CACHE.get(key)
    if cached:
        return [q] + cached

    if not settings.OPENAI_API_KEY:
        return [q]

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    prompt = f"""Rewrite the user question into {k} concise search queries.
Rules:
- Preserve all key entities and constraints.
- Preserve negations (not/never/no) and relational words (between/versus/against).
- You may use general synonyms or entailments (e.g., trade-offs, impacts, downsides) if they help retrieval.
- Do not add domain-specific terms that are not present in the question.
- Do not answer the question.
- Return JSON only: {{"queries":["...","..."]}}

Question:
{q}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You rewrite questions into search queries. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=140,
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json_object(raw)
        queries: List[str] = []
        if isinstance(parsed, dict):
            qs = parsed.get("queries")
            if isinstance(qs, list):
                for item in qs:
                    s = str(item).strip()
                    if s:
                        queries.append(s)
        if queries:
            # remove duplicates and original
            deduped = []
            seen = set()
            for s in queries:
                key_s = s.lower()
                if key_s == key or key_s in seen:
                    continue
                seen.add(key_s)
                deduped.append(s)
            _REWRITE_CACHE[key] = deduped
            return [q] + deduped
    except Exception:
        return [q]

    return [q]


def _is_yesno_question(question: str) -> bool:
    return bool(_YESNO_RE.match(question or ""))


def select_llm_hits(hits: List[Dict[str, Any]], max_hits: int = MAX_HITS_FOR_LLM) -> List[Dict[str, Any]]:
    """
    Keep a bounded citation-ID space for reliable [n] alignment.
    """
    out = [dict(h) for h in (hits or [])[:max_hits]]
    for i, h in enumerate(out, start=1):
        h["rank"] = i
    return out


def build_context(hits: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    used = 0
    for h in hits:
        label = f"[{h['rank']}] {h['source']}"
        if h.get("page_num"):
            label += f" p.{h['page_num']}"
        label += f" chunk {h['chunk_index']}"
        block = f"{label}\n{_clip(h.get('text', ''), MAX_CHARS_PER_HIT)}".strip()
        size = len(block) + 2
        if used + size > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += size
    return "\n\n".join(blocks)


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    txt = (raw or "").strip()
    if not txt:
        return None
    if txt == NO_ANSWER:
        return {"final_answer": NO_ANSWER, "claims": []}

    # strip fenced code blocks if present
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE).strip()
    txt = re.sub(r"\s*```$", "", txt).strip()

    # direct parse
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # fallback: slice between first "{" and last "}"
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(txt[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _claim_supported(claim: str, citation_ids: List[int], hits: List[Dict[str, Any]]) -> bool:
    claim_toks = list(dict.fromkeys(_tokens(claim)))
    if len(claim_toks) < 2:
        return True

    evidence = " ".join(
        hits[i - 1].get("text", "")
        for i in citation_ids
        if 1 <= i <= len(hits)
    )
    if not evidence:
        return False

    ev_toks = set(_tokens(evidence))
    if not ev_toks:
        return False

    overlap = sum(1 for t in claim_toks if t in ev_toks) / max(1, len(claim_toks))
    if overlap >= 0.28:
        return True

    # Secondary acceptance for short, concrete claims.
    if len(claim_toks) <= 4 and overlap >= 0.2:
        return True

    return False


def _validate_payload(payload: Optional[Dict[str, Any]], hits: List[Dict[str, Any]], question: str) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    final_answer = str(payload.get("final_answer", "")).strip()
    if not final_answer:
        return None
    if final_answer == NO_ANSWER:
        return {"final_answer": NO_ANSWER, "claims": []}

    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return None

    if _is_yesno_question(question) and not re.match(r"^\s*(yes|no)\b", final_answer, flags=re.IGNORECASE):
        return None

    normalized_claims: List[Dict[str, Any]] = []
    for claim_obj in claims[:MAX_CLAIMS]:
        if not isinstance(claim_obj, dict):
            return None
        claim = str(claim_obj.get("claim", "")).strip()
        citations_raw = claim_obj.get("citations")
        if not claim or not isinstance(citations_raw, list):
            return None

        citation_ids: List[int] = []
        for c in citations_raw:
            if isinstance(c, int):
                citation_ids.append(c)
            elif isinstance(c, str) and c.isdigit():
                citation_ids.append(int(c))

        citation_ids = sorted(set(citation_ids))
        if not citation_ids:
            return None
        if any(c < 1 or c > len(hits) for c in citation_ids):
            return None
        if not _claim_supported(claim, citation_ids, hits):
            return None

        normalized_claims.append({"claim": claim, "citations": citation_ids})

    if not normalized_claims:
        return None

    return {"final_answer": final_answer, "claims": normalized_claims}


def _strip_citations(text: str) -> str:
    return _CITE_RE.sub("", text or "").strip()


def _cap_citations_by_answer(payload: Dict[str, Any], hits: List[Dict[str, Any]]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not hits:
        return payload, []

    final_answer = payload.get("final_answer", "")
    if final_answer == NO_ANSWER:
        return payload, []

    claims = payload.get("claims", [])
    cited_ids = {
        int(c)
        for cl in claims
        if isinstance(cl, dict)
        for c in cl.get("citations", [])
        if isinstance(c, int)
    }
    if not cited_ids:
        return {"final_answer": NO_ANSWER, "claims": []}, []

    # Keep retrieval order; only filter to ids actually cited by claims.
    kept_idxs: List[int] = []
    rank_map: Dict[int, int] = {}
    for idx, h in enumerate(hits):
        old_rank = int(h.get("rank", idx + 1))
        if old_rank not in cited_ids:
            continue
        rank_map[old_rank] = len(kept_idxs) + 1
        kept_idxs.append(idx)

    if not kept_idxs:
        return {"final_answer": NO_ANSWER, "claims": []}, []

    new_hits: List[Dict[str, Any]] = [dict(hits[idx]) for idx in kept_idxs]
    for i, h in enumerate(new_hits, start=1):
        h["rank"] = i

    new_claims: List[Dict[str, Any]] = []
    for cl in claims:
        if not isinstance(cl, dict):
            continue
        cite_ids = cl.get("citations", [])
        if not isinstance(cite_ids, list):
            continue
        mapped = [rank_map[c] for c in cite_ids if c in rank_map]
        mapped = sorted(set(mapped))
        if mapped:
            new_claims.append({"claim": cl.get("claim", ""), "citations": mapped})

    if not new_claims:
        return {"final_answer": NO_ANSWER, "claims": []}, []

    cleaned_answer = _strip_citations(final_answer)
    return {"final_answer": cleaned_answer, "claims": new_claims}, new_hits


def _render_answer(payload: Dict[str, Any]) -> str:
    final_answer = payload["final_answer"].strip()
    if final_answer == NO_ANSWER:
        return NO_ANSWER

    claims = payload.get("claims", [])
    all_cites = sorted({c for cl in claims for c in cl.get("citations", [])})
    if all_cites and not _CITE_RE.search(final_answer):
        final_answer = f"{final_answer} " + "".join(f"[{i}]" for i in all_cites)

    lines = [final_answer]
    for cl in claims:
        text = cl.get("claim", "").strip()
        if not text:
            continue
        if text.lower() in final_answer.lower():
            continue
        cite = "".join(f"[{i}]" for i in cl.get("citations", []))
        lines.append(f"- {text} {cite}".strip())
    return "\n".join(lines)


def synthesize_answer(
    question: str,
    hits: List[Dict[str, Any]],
    max_tokens: int = 250,
) -> tuple[str, List[Dict[str, Any]]]:
    if not hits:
        return "", []

    cache_key = make_cache_key(question, hits)
    cached = get_cached_answer(cache_key)
    if cached:
        cached_obj = _extract_json_object(cached)
        valid = _validate_payload(cached_obj, hits, question)
        if valid:
            max_hits = MAX_HITS_FOR_LLM
            capped_payload, capped_hits = _cap_citations_by_answer(valid, hits)
            answer = _render_answer(capped_payload)
            return answer, capped_hits

    if not settings.OPENAI_API_KEY:
        return "LLM mode is not configured (missing OPENAI_API_KEY).", []

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    context = build_context(hits)
    is_yesno = _is_yesno_question(question)

    prompt = f"""Context:
{context}

Question:
{question}

Return JSON with this exact schema:
{{
  "final_answer": "string",
  "claims": [
    {{
      "claim": "single grounded claim",
      "citations": [1, 2]
    }}
  ]
}}

Rules:
- Use only the context.
- Every claim must cite at least one source id.
- Do not use citation ids outside 1..{len(hits)}.
- {"final_answer must start with Yes or No for this question." if is_yesno else "Keep final_answer concise and factual."}
- If evidence is insufficient, return:
  {{"final_answer":"{NO_ANSWER}","claims":[]}}
- Output JSON only, no markdown.
"""

    def _generate(user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    answer = NO_ANSWER
    retry_hints = [
        "",
        "\n\nReminder: output strict JSON only and every claim must cite source ids in range.",
    ]
    for hint in retry_hints:
        raw = _generate(prompt + hint)
        parsed = _extract_json_object(raw)
        valid = _validate_payload(parsed, hits, question)
        if valid:
            max_hits = MAX_HITS_FOR_LLM
            capped_payload, capped_hits = _cap_citations_by_answer(valid, hits)
            answer = _render_answer(capped_payload)
            set_cached_answer(cache_key, json.dumps(valid))
            return answer, capped_hits

    set_cached_answer(cache_key, json.dumps({"final_answer": NO_ANSWER, "claims": []}))
    return answer, []
