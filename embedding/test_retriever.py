from retriever import Retriever

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
