import requests
from llm.prompt_builder import build_system_prompt, build_user_prompt
from embedding.retriever import Retriever

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b-instruct"   # keep your model name

def call_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama server is not running. Start it with: ollama serve"
    except requests.exceptions.HTTPError as e:
        return f"ERROR: Ollama HTTP error: {e} | Response: {r.text}"

    data = r.json()
    # Ollama /api/chat returns {"message":{"role":"assistant","content":"..."}}
    return (data.get("message", {}) or {}).get("content", "").strip()


def main():
    # Retriever (your FAISS + metadata + embedding model)
    retriever = Retriever(
        index_path="vectorstore/faiss.index",
        meta_path="vectorstore/metadata.json",
        model_path="models/all-MiniLM-L6-v2",
        use_rerank=True
    )

    system_prompt = build_system_prompt()

    while True:
        q = input("\nEnter question (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        # ✅ Correct call: your Retriever.retrieve(query, top_k, fetch_k)
        contexts = retriever.retrieve(query=q, top_k=5, fetch_k=12)

        user_prompt = build_user_prompt(q, contexts)

        print("\n================ SYSTEM PROMPT ================\n")
        print(system_prompt)
        print("\n================ USER PROMPT (WITH CONTEXT) ================\n")
        print(user_prompt)

        print("\n================ LLM ANSWER ================\n")
        answer = call_ollama(system_prompt, user_prompt)
        print(answer)


if __name__ == "__main__":
    main()



