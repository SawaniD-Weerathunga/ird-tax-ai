# api/main.py
import os
import shutil
import subprocess
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from embedding.retriever import Retriever
from llm.guardrails import classify_retrieval
from llm.prompt_builder import build_system_prompt, build_user_prompt

PDF_DIR = "data/pdfs"

# ---- Ollama settings (can change without code edits)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # change if you pulled a different model
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

app = FastAPI(title="IRD Tax AI API", version="1.0.0")


# -----------------------------
# Request/Response models
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_llm: bool = True  # allow disabling LLM from Swagger if needed


class SourceItem(BaseModel):
    document: str
    page_range: List[int]
    section: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    status: str           # ok | missing | ambiguous
    used_llm: bool = False


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)


def run_pipeline_rebuild():
    cmds = [
        ["python", "ingestion/step3_extract_pdfs.py"],
        ["python", "ingestion/step4_chunker.py"],
        ["python", "embedding/step5_build_faiss_local.py"],
    ]
    for c in cmds:
        p = subprocess.run(c, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Pipeline failed on: {' '.join(c)}\n{p.stderr}")


def safe_extractive_answer(contexts: List[Dict[str, Any]]) -> str:
    """
    No-LLM fallback: return a short snippet from best chunk (purely extractive).
    """
    if not contexts:
        return "This information is not available in the provided IRD documents."

    best = contexts[0]
    text = (best.get("content") or "").strip()
    if not text:
        return "This information is not available in the provided IRD documents."

    snippet = text.replace("\n", " ")
    if len(snippet) > 350:
        snippet = snippet[:350].rstrip() + "..."
    return snippet


def call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Calls Ollama /api/chat
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    # Ollama chat response format:
    # { message: { role: "assistant", content: "..." }, ... }
    msg = data.get("message", {}) or {}
    return (msg.get("content") or "").strip()


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "IRD Tax AI API running"}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    ensure_dirs()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved = []
    for f in files:
        name = f.filename or ""
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Not a PDF: {name}")

        dest = os.path.join(PDF_DIR, name)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)

    try:
        run_pipeline_rebuild()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Uploaded and rebuilt index successfully", "files": saved}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty")

    retriever = Retriever(
        index_path="vectorstore/faiss.index",
        meta_path="vectorstore/metadata.json",
        model_path="models/all-MiniLM-L6-v2",
        use_rerank=True,
    )

    contexts = retriever.retrieve(query=q, top_k=req.top_k, fetch_k=max(12, req.top_k))

    # Step 9 guardrails
    status, msg = classify_retrieval(contexts)
    if status in {"missing", "ambiguous"}:
        return QueryResponse(answer=msg, sources=[], status=status, used_llm=False)

    # build sources
    sources = [
        SourceItem(
            document=c.get("document", ""),
            page_range=c.get("page_range") or [],
            section=c.get("section"),
        )
        for c in contexts[: req.top_k]
    ]

    # Step 8: LLM answer (optional)
    if req.use_llm:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(q, contexts)

        try:
            answer = call_ollama_chat(system_prompt, user_prompt)
            if answer:
                return QueryResponse(answer=answer, sources=sources, status="ok", used_llm=True)
        except requests.HTTPError as e:
            # model missing / endpoint issues => fallback
            pass
        except Exception:
            pass

    # fallback: extractive answer
    answer = safe_extractive_answer(contexts)
    return QueryResponse(answer=answer, sources=sources, status="ok", used_llm=False)
