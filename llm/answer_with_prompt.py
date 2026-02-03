from llm.system_prompt import SYSTEM_PROMPT
from llm.prompt_builder import build_user_prompt

# ✅ import your retriever function (adjust to your project)
from embedding.test_retriever import retrieve_top_chunks


def main():
    while True:
        q = input("\nEnter question (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        # Step 6: retrieve top chunks
        contexts = retrieve_top_chunks(q, top_k=5)

        # Step 7: build prompts
        user_prompt = build_user_prompt(q, contexts)

        print("\n================ SYSTEM PROMPT ================\n")
        print(SYSTEM_PROMPT)

        print("\n================ USER PROMPT (WITH CONTEXT) ================\n")
        print(user_prompt)

        # ✅ Step 7 ends here (safe prompt + context ready)
        # Step 8 will call an LLM to generate the answer.


if __name__ == "__main__":
    main()
