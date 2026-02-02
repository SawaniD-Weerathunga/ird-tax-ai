import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vectorstore/faiss.index"
META_PATH  = "vectorstore/metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def main():
    # Load model
    model = SentenceTransformer(MODEL_NAME)

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

    # Search
    k = 5
    scores, ids = index.search(qvec, k)

    print("\nTop matches:\n")
    for rank, idx in enumerate(ids[0], start=1):
        item = meta[idx]
        print(f"{rank}) score={scores[0][rank-1]:.4f}")
        print(f"   doc: {item['document']}")
        print(f"   pages: {item.get('page_range')}")
        print(f"   section: {item.get('section')}")
        print(f"   preview: {item['content'][:220]}...")
        print()

if __name__ == "__main__":
    main()
