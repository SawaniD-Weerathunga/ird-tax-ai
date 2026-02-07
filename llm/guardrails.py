import os
from typing import List, Dict, Tuple

DEFAULT_UNAVAILABLE_MSG = "This information is not available in the provided IRD documents."


def get_score(item: Dict) -> float:
    # works with reranked results OR faiss-only results
    if "final_score" in item:
        return float(item["final_score"])
    return float(item.get("score", 0.0))


def _get_float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def classify_retrieval(
    contexts: List[Dict],
    min_score: float | None = None,
    min_gap: float | None = None,
    min_top2_score: float | None = None,
) -> Tuple[str, str]:
    """
    Returns (status, message)
    status: "ok" | "missing" | "ambiguous"
    """
    if min_score is None:
        min_score = _get_float_env("RETRIEVAL_MIN_SCORE", 0.55)
    if min_gap is None:
        min_gap = _get_float_env("RETRIEVAL_MIN_GAP", 0.10)
    if min_top2_score is None:
        min_top2_score = _get_float_env("RETRIEVAL_MIN_TOP2", 0.80)

    if not contexts:
        return "missing", DEFAULT_UNAVAILABLE_MSG

    # sort by score desc (safe)
    contexts = sorted(contexts, key=get_score, reverse=True)
    s1 = get_score(contexts[0])

    # Missing: top result too weak
    if s1 < min_score:
        return "missing", DEFAULT_UNAVAILABLE_MSG

    # Ambiguous: top1 and top2 very close and both decent
    if len(contexts) >= 2:
        s2 = get_score(contexts[1])
        gap = s1 - s2
        if s2 >= min_top2_score and gap < min_gap:
            msg = (
                "Your question may be ambiguous. "
                "Do you mean Personal Income Tax (PIT), Corporate Income Tax (CIT), or SET instructions? "
                "Please specify the tax type and year of assessment."
            )
            return "ambiguous", msg

    return "ok", ""
