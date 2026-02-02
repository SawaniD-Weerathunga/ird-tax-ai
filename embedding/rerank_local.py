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

def doc_priority(doc_name: str) -> int:
    """
    Higher = better.
    Tune this based on your project needs.
    """
    d = doc_name.lower()

    # PN notice should win for amendment/relief questions
    if d.startswith("pn_") or "pn_it" in d or "public notice" in d:
        return 5

    # SET guide is good but not always "official change notice"
    if "set_" in d or "statement of estimated tax" in d:
        return 3

    # CIT guide generally lower priority for personal relief
    if "cit" in d or "asmt_cit" in d:
        return 2

    return 1

def keyword_score(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))

    if not q_tokens:
        return 0.0

    overlap = len(q_tokens & t_tokens) / len(q_tokens)

    # extra boosts for important phrases
    boosts = 0.0
    q_low = query.lower()
    t_low = text.lower()

    if "personal relief" in q_low and "personal relief" in t_low:
        boosts += 0.35
    if "2025/2026" in q_low and "2025/2026" in t_low:
        boosts += 0.25
    if "1,800,000" in t_low or "1800000" in t_low:
        boosts += 0.10
    if "rs." in t_low or "lkr" in t_low:
        boosts += 0.05

    return overlap + boosts

def rerank(query: str, candidates: List[Dict], alpha: float = 1.0, beta: float = 0.6, gamma: float = 0.25) -> List[Tuple[float, Dict]]:
    """
    candidates must include:
      - 'score' (faiss similarity you printed)
      - 'document'
      - 'content' or 'preview'
    We combine:
      final = alpha*faiss + beta*keyword + gamma*doc_priority
    """
    reranked = []
    for c in candidates:
        faiss_sim = float(c.get("score", 0.0))
        text = c.get("content") or c.get("preview") or ""
        kw = keyword_score(query, text)
        pr = doc_priority(c.get("document", ""))

        final = alpha * faiss_sim + beta * kw + gamma * pr
        reranked.append((final, c))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked
