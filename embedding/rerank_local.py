import re
from typing import List, Dict, Tuple

STOPWORDS = {
    "the","is","a","an","of","to","and","or","in","on","for","with","as","by",
    "what","which","who","when","where","how","why","from","are","was","were",
    "be","been","being","this","that","these","those"
}

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9/.,% ]+", " ", text)
    toks = [t for t in text.split() if t and t not in STOPWORDS]
    return toks

def doc_priority(doc_name: str) -> float:
    """
    Return a SMALL float boost (0.0 to ~0.6).
    This avoids overpowering semantic relevance.
    """
    d = (doc_name or "").lower()

    # Highest priority: Public Notices
    if d.startswith("pn_") or "pn_it" in d or "public notice" in d:
        return 0.60

    # Guides / instructions
    if "set_" in d or "statement of estimated tax" in d:
        return 0.25

    if "cit" in d or "asmt_cit" in d:
        return 0.15

    return 0.0

def keyword_score(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))

    if not q_tokens:
        return 0.0

    overlap = len(q_tokens & t_tokens) / len(q_tokens)

    boosts = 0.0
    q_low = query.lower()
    t_low = (text or "").lower()

    # Tax-type synonym boosts
    if ("corporate income tax" in q_low or "corporate tax" in q_low or "cit" in q_low) and (
        "company tax" in t_low or "corporate" in t_low or "cit" in t_low
    ):
        boosts += 0.20
    if ("self employment tax" in q_low or "set" in q_low) and (
        "statement of estimated tax" in t_low or "estimated tax" in t_low or "set" in t_low
    ):
        boosts += 0.20
    if ("personal income tax" in q_low or "pit" in q_low) and (
        "personal income tax" in t_low or "pit" in t_low
    ):
        boosts += 0.15

    # Year-of-assessment boost
    for y in re.findall(r"\b20\d{2}\s*/\s*20\d{2}\b", q_low):
        if y.replace(" ", "") in t_low.replace(" ", ""):
            boosts += 0.30

    if "personal relief" in q_low and "personal relief" in t_low:
        boosts += 0.35
    if "2025/2026" in q_low and "2025/2026" in t_low:
        boosts += 0.25
    if "1,800,000" in t_low or "1800000" in t_low:
        boosts += 0.10
    if "rs." in t_low or "lkr" in t_low:
        boosts += 0.05

    return overlap + boosts

def section_boost(query: str, section: str) -> float:
    """
    Small boost if query keywords appear in section/title.
    """
    if not section:
        return 0.0
    return 0.25 * keyword_score(query, section)

def rerank(
    query: str,
    candidates: List[Dict],
    alpha: float = 1.0,
    beta: float = 0.6,
    gamma: float = 1.2,
    delta: float = 0.25
) -> List[Tuple[float, Dict]]:
    """
    candidates must include:
      - 'score' (faiss similarity)
      - 'document'
      - 'content' or 'preview'
      - optional 'section'

    final = alpha*faiss + beta*keyword + gamma*doc_priority + delta*section_boost
    """
    reranked = []
    for c in candidates:
        faiss_sim = float(c.get("score", 0.0))
        text = c.get("content") or c.get("preview") or ""
        section = c.get("section", "")

        kw = keyword_score(query, text)
        pr = doc_priority(c.get("document", ""))
        sb = section_boost(query, section)

        final = alpha * faiss_sim + beta * kw + gamma * pr + delta * sb
        reranked.append((final, c))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked
