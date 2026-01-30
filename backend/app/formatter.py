# app/formatter.py
import re
from typing import List, Dict, Any, Optional

STRONG_SCORE = 0.55
MAX_HITS_WEAK = 2
MAX_SNIPPET_CHARS = 380
MAX_ANSWER_CHARS = 380


def _clean(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _label(hit: Dict[str, Any]) -> str:
    src = hit.get("source", "unknown")
    p = hit.get("page_num")
    ck = hit.get("chunk_index")
    return f"{src} p.{p} — chunk {ck}" if p else f"{src} — chunk {ck}"


def _first_sentences(text: str, n: int = 2) -> str:
    t = _clean(text)
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    snippet = " ".join(parts[:n]).strip()
    if len(snippet) > MAX_SNIPPET_CHARS:
        snippet = snippet[:MAX_SNIPPET_CHARS].rstrip() + "..."
    return snippet


def _infer_question_type(q: str) -> str:
    ql = (q or "").strip().lower()
    # treat "what is/does/stand for/define" as definition intent
    if any(k in ql for k in ["what is", "what does", "stand for", "define", "what does it mean"]):
        return "definition"
    if any(k in ql for k in ["list", "what are", "components", "requirements", "features"]):
        return "list"
    return "generic"


def _extract_term(q: str) -> str:
    """
    Pull the likely target term from common question patterns.
    Examples:
      - "What is p95 latency?" -> "p95 latency"
      - "What does RAG stand for and what does it mean?" -> "RAG"
      - "What is FAISS used for in this project?" -> "FAISS"
    """
    q = (q or "").strip()

    m = re.search(r"(?i)\bwhat\s+is\s+(.+?)\??$", q)
    if m:
        term = m.group(1)
        term = re.sub(r"(?i)\bused\s+for.*$", "", term).strip()
        term = re.sub(r"(?i)\bin\s+this\s+project.*$", "", term).strip()
        return term.strip(' "“”')

    m = re.search(r"(?i)\bwhat\s+does\s+(.+?)\s+stand\s+for\b", q)
    if m:
        return m.group(1).strip(' "“”')

    m = re.search(r"(?i)\bdefine\s+(.+?)\??$", q)
    if m:
        return m.group(1).strip(' "“”')

    # fallback: nothing extracted
    return ""


def _choose_best_hit_for_term(hits: List[Dict[str, Any]], term: str) -> Dict[str, Any]:
    """
    Prefer hits where the term actually appears in the text,
    with a small extra boost if source name suggests glossary.
    """
    term_l = (term or "").lower()

    def score_hit(h):
        base = float(h.get("score", 0.0))
        text = (h.get("text") or "").lower()
        src = (h.get("source") or "").lower()
        contains = 1.0 if term_l and term_l in text else 0.0
        glossary_boost = 0.15 if "glossary" in src else 0.0
        return base + (0.35 * contains) + glossary_boost

    return max(hits, key=score_hit)


def _extract_section_for_term(text: str, term: str) -> str:
    """
    Term-aware extraction inside a chunk that contains multiple glossary entries.
    Handles patterns like:
      ## p95 latency
      p95 (...) ...
      ## p50 latency
    Also handles inline:
      RAG Retrieval-Augmented Generation: ...
    """
    t = _clean(text)
    if not t or not term:
        return ""

    term_esc = re.escape(term)

    # 1) Markdown-style heading sections
    pat_md = rf"(?is)^\s*##\s*{term_esc}\s*(.*?)(?=^\s*##\s+|\Z)"
    m = re.search(pat_md, t, flags=re.MULTILINE)
    if m:
        body = _clean(m.group(1))
        return _first_sentences(body, n=2)

    # 2) Inline definition lines: "TERM: ...." or "TERM - ...."
    pat_inline = rf"(?is)\b{term_esc}\b\s*[:\-–]\s*(.+)"
    m = re.search(pat_inline, t)
    if m:
        return _first_sentences(m.group(1), n=2)

    # 3) Glossary style: line contains term then definition continues on same line
    # e.g. "## RAG Retrieval-Augmented Generation: retrieve ..."
    pat_gloss = rf"(?is)##\s*{term_esc}\s+(.+?)(?=##|\Z)"
    m = re.search(pat_gloss, t)
    if m:
        return _first_sentences(m.group(1), n=2)

    # 4) Fallback: find first occurrence of term and grab nearby text
    idx = t.lower().find(term.lower())
    if idx != -1:
        window = t[idx : idx + 600]
        return _first_sentences(window, n=2)

    return ""

def format_answer(hits: List[Dict[str, Any]], question: Optional[str] = None) -> str:
    if not hits:
        return ""

    q = (question or "").strip()
    hits = sorted(hits, key=lambda h: float(h.get("score", 0.0)), reverse=True)

    top_score = float(hits[0].get("score", 0.0))
    k = 1 if top_score >= STRONG_SCORE else min(MAX_HITS_WEAK, len(hits))

    qtype = _infer_question_type(q)

    # --- choose answer_hit (the hit we actually used to create the Answer line) ---
    if qtype == "definition":
        term = _extract_term(q)
        answer_hit = _choose_best_hit_for_term(hits, term) if term else hits[0]
        answer_line = _extract_section_for_term(answer_hit.get("text", ""), term) if term else _first_sentences(answer_hit.get("text", ""), 2)
    else:
        answer_hit = hits[0]
        answer_line = _first_sentences(answer_hit.get("text", ""), 2)

    answer_line = _clean(answer_line)
    if len(answer_line) > MAX_ANSWER_CHARS:
        answer_line = answer_line[:MAX_ANSWER_CHARS].rstrip() + "..."

    # --- Evidence: always include answer_hit first, then fill with other top hits ---
    evidence_hits = [answer_hit]
    for h in hits:
        if len(evidence_hits) >= k:
            break
        if h is answer_hit:
            continue
        # de-dupe by (source, page_num, chunk_index)
        if any(
            (h.get("source"), h.get("page_num"), h.get("chunk_index"))
            == (eh.get("source"), eh.get("page_num"), eh.get("chunk_index"))
            for eh in evidence_hits
        ):
            continue
        evidence_hits.append(h)

    evidence_lines = []
    for h in evidence_hits:
        if qtype == "definition":
            term = _extract_term(q)
            snippet = _extract_section_for_term(h.get("text", ""), term) or _first_sentences(h.get("text", ""), 2)
        else:
            snippet = _first_sentences(h.get("text", ""), 2)
        evidence_lines.append(f"- ({_label(h)}) {snippet}")

    out = "Answer: " + (answer_line if answer_line else "See evidence below.")
    out += "\n\nEvidence:\n" + "\n".join(evidence_lines)
    return out
