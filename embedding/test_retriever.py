from embedding.retriever import Retriever

def main():
    retriever = Retriever(use_rerank=True)  # True = PN notices get prioritized

    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break

        results = retriever.retrieve(q, top_k=5, fetch_k=12)

        print("\nTop results:\n")
        for i, r in enumerate(results, start=1):
            score = r.get("final_score", r.get("score"))
            print(f"{i}) score={score:.4f}")
            print(f"   doc: {r['document']}")
            print(f"   pages: {r['page_range']}")
            print(f"   section: {r.get('section')}")
            print(f"   preview: {r['content'][:220].replace('\\n',' ')}...")
            print()

if __name__ == "__main__":
    main()

def retrieve_top_chunks(query: str, top_k: int = 5):
    """
    Reuse the same logic your test_retriever uses.
    Must return list of dicts with: document, page_range, section, content, score
    """
    # Import inside to avoid circular import issues
    # If you already have functions for search+rerank, call them here.

    # ---- Example skeleton (adjust names to your file) ----
    import json
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from embedding.rerank_local import rerank

    INDEX_PATH = "vectorstore/faiss.index"
    META_PATH = "vectorstore/metadata.json"

    def l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    model = SentenceTransformer("models/all-MiniLM-L6-v2")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    qvec = model.encode([query], convert_to_numpy=True).astype("float32")
    qvec = l2_normalize(qvec)

    k = max(top_k, 8)  # search a bit more, then rerank
    scores, ids = index.search(qvec, k)

    candidates = []
    for rank, idx in enumerate(ids[0], start=1):
        item = meta[idx]
        candidates.append({
            "score": float(scores[0][rank-1]),
            "document": item.get("document"),
            "page_range": item.get("page_range"),
            "section": item.get("section"),
            "content": item.get("content", "")
        })

    reranked = rerank(query, candidates)
    results = []
    for final_score, item in reranked[:top_k]:
        item2 = item.copy()
        item2["score"] = float(final_score)
        results.append(item2)

    return results
