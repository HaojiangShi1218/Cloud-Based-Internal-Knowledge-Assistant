# app/formatter.py
import re
from typing import List, Dict, Any, Optional

STRONG_SCORE = 0.55
MAX_HITS_WEAK = 2
MAX_SNIPPET_CHARS = 2000
MAX_ANSWER_CHARS = 1500
MAX_VISIBLE_LLM_EVIDENCE = 3


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
    pe = hit.get("page_end", p)
    if p is None:
        return f"Source: {src}"
    if pe is not None and pe != p:
        return f"Source: {src}, Pages: {p}-{pe}"
    return f"Source: {src}, Page: {p}"

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


def _select_evidence_hits(hits: List[Dict[str, Any]], question: Optional[str] = None) -> List[Dict[str, Any]]:
    if not hits:
        return []

    q = (question or "").strip()
    hits = sorted(hits, key=lambda h: float(h.get("final_score", h.get("score", 0.0))), reverse=True)

    top_score = float(hits[0].get("score", 0.0))
    k = 1 if top_score >= STRONG_SCORE else min(MAX_HITS_WEAK, len(hits))

    qtype = _infer_question_type(q)

    if qtype == "definition":
        term = _extract_term(q)
        answer_hit = _choose_best_hit_for_term(hits, term) if term else hits[0]
    else:
        answer_hit = hits[0]

    evidence_hits = [answer_hit]
    for h in hits:
        if len(evidence_hits) >= k:
            break
        if h is answer_hit:
            continue
        if any(
            (h.get("source"), h.get("page_num"), h.get("chunk_index"))
            == (eh.get("source"), eh.get("page_num"), eh.get("chunk_index"))
            for eh in evidence_hits
        ):
            continue
        evidence_hits.append(h)

    return evidence_hits


def _build_evidence_lines(hits: List[Dict[str, Any]], question: Optional[str] = None) -> List[str]:
    q = (question or "").strip()
    qtype = _infer_question_type(q)
    evidence_lines = []
    for h in hits:
        if qtype == "definition":
            term = _extract_term(q)
            snippet = _extract_section_for_term(h.get("text", ""), term) or _first_sentences(h.get("text", ""), 10)
        else:
            snippet = _first_sentences(h.get("text", ""), 10)
        evidence_lines.append(f"- ({_label(h)}) {snippet}")
    return evidence_lines


def format_answer_with_evidence(
    answer_text: str,
    hits: List[Dict[str, Any]],
    question: Optional[str] = None,
    use_all_hits: bool = False,
) -> str:
    answer_text = _clean(answer_text) or "See relevant evidence below."
    if not hits:
        return f"Answer: {answer_text}"

    evidence_hits = hits if use_all_hits else _select_evidence_hits(hits, question=question)
    if use_all_hits:
        evidence_hits = evidence_hits[:MAX_VISIBLE_LLM_EVIDENCE]
    evidence_lines = _build_evidence_lines(evidence_hits, question=question)

    out = f"Answer: {answer_text}"
    out += "\n\nEvidence:\n" + "\n".join(evidence_lines)
    return out

def format_answer(hits: List[Dict[str, Any]], question: Optional[str] = None) -> str:
    if not hits:
        return "No evidence found"
    return format_answer_with_evidence("See relevant evidence below.", hits, question=question, use_all_hits=False)
