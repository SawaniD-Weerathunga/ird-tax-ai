import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_ollama(prompt: str, model: str = "llama3.2:3b", temperature: float = 0.2) -> str:
    """
    Calls local Ollama model and returns the generated text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()
