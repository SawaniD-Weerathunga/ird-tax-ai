import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def ollama_is_up(timeout: int = 3) -> bool:
    """
    Returns True if Ollama server is reachable.
    """
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def call_ollama_chat(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    """
    Uses Ollama /api/chat endpoint (recommended).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }

    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)

    # Better error message
    if r.status_code != 200:
        try:
            return f"ERROR: Ollama HTTP {r.status_code}: {r.text}"
        except Exception:
            return f"ERROR: Ollama HTTP {r.status_code}"

    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()
