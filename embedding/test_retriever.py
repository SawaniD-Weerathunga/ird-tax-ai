from embedding.retriever import Retriever
from llm.guardrails import classify_retrieval

# Create ONE retriever instance (so model + FAISS don't reload repeatedly)
retriever = Retriever(
    index_path="vectorstore/faiss.index",
    meta_path="vectorstore/metadata.json",
    model_path="models/all-MiniLM-L6-v2",
    use_rerank=True
)

def retrieve_top_chunks(query: str, top_k: int = 5, fetch_k: int = 12):
    """
    Reusable function (used later by LLM scripts too).
    Returns list of dicts with: document, page_range, section, content, score/final_score.
    """
    return retriever.retrieve(query=query, top_k=top_k, fetch_k=fetch_k)

def main():
    while True:
        q = input("\nEnter query (or 'exit'): ").strip()

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # Step 6: retrieve
        results = retrieve_top_chunks(q, top_k=5, fetch_k=12)

        # ✅ Step 9: Guardrails (missing / ambiguous)
        status, msg = classify_retrieval(results)

        if status in {"missing", "ambiguous"}:
            print("\n" + msg)
            continue

        # status == "ok" -> print results like before
        print("\nTop results:\n")
        for i, r in enumerate(results, start=1):
            score = r.get("final_score", r.get("score", 0.0))
            print(f"{i}) score={score:.4f}")
            print(f"   doc: {r.get('document')}")
            print(f"   pages: {r.get('page_range')}")
            print(f"   section: {r.get('section')}")
            preview = (r.get("content") or "")[:220].replace("\n", " ")
            print(f"   preview: {preview}...")
            print()

if __name__ == "__main__":
    main()
