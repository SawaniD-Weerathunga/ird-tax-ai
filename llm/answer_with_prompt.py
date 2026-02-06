from llm.system_prompt import SYSTEM_PROMPT
from llm.prompt_builder import build_user_prompt
from llm.guardrails import classify_retrieval
from llm.disclaimer import append_disclaimer

from embedding.test_retriever import retrieve_top_chunks


def main():
    while True:
        q = input("\nEnter question (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # Step 6: retrieve top chunks
        contexts = retrieve_top_chunks(q, top_k=5)

        # Step 9: guardrails (missing / ambiguous)
        status, msg = classify_retrieval(contexts)
        if status in {"missing", "ambiguous"}:
            print("\n" + append_disclaimer(msg))
            continue

        # Step 7: build prompts (safe prompt + context)
        user_prompt = build_user_prompt(q, contexts)

        print("\n================ SYSTEM PROMPT ================\n")
        print(SYSTEM_PROMPT)

        print("\n================ USER PROMPT (WITH CONTEXT) ================\n")
        print(user_prompt)

        # Step 11: show disclaimer when printing (for this script, we print it alone)
        print("\n================ DISCLAIMER ================\n")
        print(append_disclaimer(""))


if __name__ == "__main__":
    main()
