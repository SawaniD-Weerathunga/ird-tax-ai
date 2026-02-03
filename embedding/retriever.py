import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Optional reranker (if file exists)
try:
    from rerank_local import rerank
    HAS_RERANK = True
except Exception:
    HAS_RERANK = False


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Retriever:
    """
    Retriever = Search Brain
    - Embeds query
    - Searches FAISS
    - Returns top relevant chunks (with citations)
    - Does NOT answer
    """

    def __init__(
        self,
        index_path="vectorstore/faiss.index",
        meta_path="vectorstore/metadata.json",
        model_path="models/all-MiniLM-L6-v2",
        use_rerank=True,
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_path = model_path
        self.use_rerank = use_rerank and HAS_RERANK

        # Load model (local path)
        self.model = SentenceTransformer(self.model_path)

        # Load FAISS index
        self.index = faiss.read_index(self.index_path)

        # Load metadata
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 12):
        """
        fetch_k = how many to pull from FAISS before reranking
        top_k   = final number returned
        """
        query = (query or "").strip()
        if not query:
            return []

        # Embed query
        qvec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        qvec = l2_normalize(qvec)

        # Search FAISS
        scores, ids = self.index.search(qvec, fetch_k)

        # Build candidate list
        candidates = []
        for rank, idx in enumerate(ids[0]):
            if idx < 0:
                continue
            item = self.meta[idx]
            candidates.append({
                "score": float(scores[0][rank]),
                "document": item.get("document"),
                "page_range": item.get("page_range"),
                "section": item.get("section"),
                "content": item.get("content", ""),
            })

        # Optional rerank (recommended)
        if self.use_rerank:
            reranked = rerank(query, candidates, alpha=1.0, beta=0.6, gamma=0.25, delta=1.2)
            # rerank returns list of (final_score, item)
            final = []
            for final_score, item in reranked[:top_k]:
                item2 = dict(item)
                item2["final_score"] = float(final_score)
                final.append(item2)
            return final

        # If no rerank: return FAISS top_k directly
        return candidates[:top_k]
