import os
import requests

from embedding.retriever import Retriever
from llm.prompt_builder import build_system_prompt, build_user_prompt
from llm.guardrails import classify_retrieval
from llm.disclaimer import append_disclaimer

# Ollama settings (can override without editing code)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama server is not running. Start it with: ollama serve"
    except requests.exceptions.HTTPError as e:
        # show ollama JSON error if available
        try:
            err = r.json().get("error")
        except Exception:
            err = None
        if err:
            return f"ERROR: Ollama HTTP error: {e} | {err}"
        return f"ERROR: Ollama HTTP error: {e} | Response: {getattr(r, 'text', '')}"

    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


def main():
    retriever = Retriever(
        index_path="vectorstore/faiss.index",
        meta_path="vectorstore/metadata.json",
        model_path="models/all-MiniLM-L6-v2",
        use_rerank=True,
    )

    system_prompt = build_system_prompt()

    while True:
        q = input("\nEnter question (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        # Step 6: retrieve
        contexts = retriever.retrieve(query=q, top_k=5, fetch_k=12)

        # Step 9: guardrails
        status, msg = classify_retrieval(contexts)
        if status in {"missing", "ambiguous"}:
            print("\n" + append_disclaimer(msg))
            continue

        # Step 7: build prompt
        user_prompt = build_user_prompt(q, contexts)

        print("\n================ SYSTEM PROMPT ================\n")
        print(system_prompt)
        print("\n================ USER PROMPT (WITH CONTEXT) ================\n")
        print(user_prompt)

        # Step 8: LLM answer
        print("\n================ LLM ANSWER ================\n")
        answer = call_ollama(system_prompt, user_prompt)

        # If Ollama returns an error string, still append disclaimer
        print(append_disclaimer(answer))


if __name__ == "__main__":
    main()
