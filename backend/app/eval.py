"""
Run retrieval evaluation (no LLM required).

Usage (from backend/):
    python -m app.eval
Optional env vars:
    EVAL_SET_PATH=eval_set.json
    EVAL_TOP_K=5
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger

from app.rag import retrieve


def load_eval_set(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


def hit_contains_required_text(hits: List[Dict[str, Any]], required: List[str]) -> bool:
    if not required:
        return True
    hay = "\n".join([h.get("text", "") for h in hits]).lower()
    return all(r.lower() in hay for r in required)


def expected_source_in_hits(hits: List[Dict[str, Any]], expected_sources: List[str]) -> bool:
    if not expected_sources:
        return True
    got = {h.get("source") for h in hits}
    return any(s in got for s in expected_sources)


def evaluate_case(
    question: str,
    expected_sources: List[str],
    expected_contains: List[str],
    top_k: int,
) -> Tuple[bool, List[Dict[str, Any]]]:
    hits = retrieve(question, top_k=top_k)
    ok_source = expected_source_in_hits(hits, expected_sources)
    ok_text = hit_contains_required_text(hits, expected_contains)
    return (ok_source and ok_text), hits


def main():
    eval_path = Path(os.getenv("EVAL_SET_PATH", "eval_set.json"))
    top_k = int(os.getenv("EVAL_TOP_K", "5"))

    eval_set = load_eval_set(eval_path)
    cases = eval_set.get("cases", [])

    if not cases:
        logger.error("No cases found in eval set.")
        return

    logger.info(f"Loaded eval set: {eval_set.get('name', 'Unnamed')} v{eval_set.get('version', '?')}")
    logger.info(f"Using top_k={top_k}")
    logger.info("-" * 80)

    passed = 0
    failed_cases = []

    t0 = time.time()

    for i, case in enumerate(cases, start=1):
        cid = case.get("id", f"case_{i}")
        q = case["question"]
        expected_sources = case.get("expected_sources", [])
        expected_contains = case.get("expected_contains", [])

        ok, hits = evaluate_case(q, expected_sources, expected_contains, top_k=top_k)

        if ok:
            passed += 1
            logger.success(f"[{i}/{len(cases)}] PASS  {cid}  — {q}")
        else:
            logger.error(f"[{i}/{len(cases)}] FAIL  {cid}  — {q}")
            failed_cases.append((cid, q, expected_sources, expected_contains, hits))

            # concise debug view
            top = hits[: min(3, len(hits))]
            if not top:
                logger.warning("  No hits returned.")
            else:
                logger.warning("  Top hits:")
                for h in top:
                    src = h.get("source")
                    pg = h.get("page_num")
                    sc = h.get("score")
                    logger.warning(f"   - {src} (p.{pg}) score={sc:.4f}" if pg else f"   - {src} score={sc:.4f}")

    total = len(cases)
    dt = time.time() - t0
    logger.info("-" * 80)
    logger.info(f"Result: {passed}/{total} passed  ({passed/total:.1%})   time={dt:.2f}s")

    if failed_cases:
        logger.info("\nDetailed misses:")
        for (cid, q, exp_src, exp_txt, hits) in failed_cases:
            logger.info("=" * 80)
            logger.info(f"Case: {cid}")
            logger.info(f"Q: {q}")
            logger.info(f"Expected sources: {exp_src}")
            logger.info(f"Expected contains: {exp_txt}")
            logger.info("Returned hits (top 5):")
            for h in hits[:5]:
                logger.info({
                    "rank": h.get("rank"),
                    "source": h.get("source"),
                    "page_num": h.get("page_num"),
                    "chunk_index": h.get("chunk_index"),
                    "score": h.get("score"),
                })

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
