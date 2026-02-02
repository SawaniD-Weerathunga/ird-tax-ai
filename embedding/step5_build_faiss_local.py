import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -------- Paths --------
INPUT_CHUNKS = "data/chunks/ird_chunks.json"
OUT_DIR = "vectorstore"
INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")
META_PATH  = os.path.join(OUT_DIR, "metadata.json")
CONFIG_PATH = os.path.join(OUT_DIR, "config.json")

os.makedirs(OUT_DIR, exist_ok=True)

# -------- Config --------
MODEL_NAME = "all-MiniLM-L6-v2"   # fast + popular + good for RAG
BATCH_SIZE = 64                  # local encoding batch size


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors so cosine similarity works with FAISS inner product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def main():
    # 1) Load chunks
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["content"] for c in chunks]
    print(f"Loaded {len(texts)} chunks")

    # 2) Load local embedding model
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 3) Create embeddings
    print("Creating embeddings locally...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype("float32")

    # 4) Normalize for cosine similarity
    vectors = l2_normalize(vectors)

    # 5) Build FAISS index (cosine similarity using inner product)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # 6) Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # 7) Save metadata (for citations)
    metadata = []
    for c in chunks:
        metadata.append({
            "chunk_id": c["chunk_id"],
            "document": c["document"],
            "page_range": c.get("page_range"),
            "section": c.get("section", "Not specified"),
            "content": c["content"]
        })

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 8) Save config
    config = {
        "embedding_model": MODEL_NAME,
        "dimension": dim,
        "index_type": "IndexFlatIP",
        "normalized": True,
        "num_vectors": len(metadata),
        "input_file": INPUT_CHUNKS,
        "batch_size": BATCH_SIZE
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 9) Print stats
    print("\n✅ STEP 5 COMPLETE (LOCAL)")
    print(f"Vectors stored: {len(metadata)}")
    print(f"Vector dim: {dim}")
    print(f"Saved FAISS index: {INDEX_PATH}")
    print(f"Saved metadata:    {META_PATH}")
    print(f"Saved config:      {CONFIG_PATH}")


if __name__ == "__main__":
    main()
