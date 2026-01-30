from typing import List, Dict, Optional
from openai import OpenAI
from app.config import settings
from app.cache import make_cache_key, get_cached_answer, set_cached_answer

SYSTEM_PROMPT = """You are a careful technical assistant.
Answer ONLY using the provided context.
If the answer is not contained in the context, say "I don’t know based on the provided documents."
Keep answers concise and factual.
"""

MAX_CONTEXT_CHARS = 6000  # MVP heuristic

def build_context(hits: List[Dict]) -> str:
    blocks = []
    for h in hits:
        label = f"[{h['rank']}] {h['source']}"
        if h.get("page_num"):
            label += f" p.{h['page_num']}"
        label += f" chunk {h['chunk_index']}"
        blocks.append(f"{label}\n{h['text']}".strip())
    joined = "\n\n".join(blocks)
    return joined[:MAX_CONTEXT_CHARS]

def synthesize_answer(question: str, hits: List[Dict], max_tokens: int = 250) -> str:
    if not hits:
        return ""

    cache_key = make_cache_key(question, hits)
    cached = get_cached_answer(cache_key)
    if cached:
        return cached

    if not settings.OPENAI_API_KEY:
        return "LLM mode is not configured (missing OPENAI_API_KEY)."

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    context = build_context(hits)
    prompt = f"""Context:
{context}

Question:
{question}

Answer:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )

    answer = (resp.choices[0].message.content or "").strip()
    set_cached_answer(cache_key, answer)
    return answer
