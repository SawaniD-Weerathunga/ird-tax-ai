from typing import List, Dict, Tuple

DEFAULT_UNAVAILABLE_MSG = "This information is not available in the provided IRD documents."

def get_score(item: Dict) -> float:
    # works with reranked results OR faiss-only results
    if "final_score" in item:
        return float(item["final_score"])
    return float(item.get("score", 0.0))

def classify_retrieval(
    contexts: List[Dict],
    min_score: float = 1.0,          # ✅ stricter threshold (recommended)
    min_gap: float = 0.10,           # ambiguity threshold
    min_top2_score: float = 0.80     # if top2 also high + close -> ambiguous
) -> Tuple[str, str]:
    """
    Returns (status, message)
    status: "ok" | "missing" | "ambiguous"
    """

    if not contexts:
        return "missing", DEFAULT_UNAVAILABLE_MSG

    # sort by score desc (safe)
    contexts = sorted(contexts, key=get_score, reverse=True)
    top1 = contexts[0]
    s1 = get_score(top1)

    # ✅ Missing: top result too weak
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