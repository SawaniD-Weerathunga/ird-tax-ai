import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rerank_local import rerank


INDEX_PATH = "vectorstore/faiss.index"
META_PATH  = "vectorstore/metadata.json"

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def main():
    # Load model (local cached path as you are using)
    model = SentenceTransformer("models/all-MiniLM-L6-v2")

    # Load FAISS + metadata
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    query = input("Enter query: ").strip()
    if not query:
        return

    # Embed query
    qvec = model.encode([query], convert_to_numpy=True).astype("float32")
    qvec = l2_normalize(qvec)

    # Search (get more candidates first, then rerank)
    k = 10
    scores, ids = index.search(qvec, k)

    # Build candidate list for reranker
    results = []
    for rank, idx in enumerate(ids[0]):
        if idx < 0:
            continue

        item = meta[idx]
        results.append({
            "score": float(scores[0][rank]),
            "document": item.get("document"),
            "page_range": item.get("page_range"),
            "section": item.get("section"),
            "content": item.get("content", ""),
            "preview": item.get("content", "")[:400].replace("\n", " ")
        })

    # Print FAISS top 5
    print("\nTop matches (FAISS):\n")
    for i, r in enumerate(results[:5], start=1):
        print(f"{i}) score={r['score']:.4f}")
        print(f"   doc: {r['document']}")
        print(f"   pages: {r['page_range']}")
        print(f"   section: {r['section']}")
        print(f"   preview: {r['preview'][:220]}...")
        print()

    # Rerank and print reranked top 5
    reranked = rerank(query, results, alpha=1.0, beta=0.6, gamma=0.25)

    print("\n✅ Top matches (RERANKED):\n")
    for i, (final, r) in enumerate(reranked[:5], start=1):
        print(f"{i}) final={final:.4f} (faiss={r['score']:.4f})")
        print(f"   doc: {r['document']}")
        print(f"   pages: {r['page_range']}")
        print(f"   section: {r['section']}")
        print(f"   preview: {r['preview'][:220]}...")
        print()

if __name__ == "__main__":
    main()
