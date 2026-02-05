# api/main.py
import os
import shutil
import subprocess
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from embedding.retriever import Retriever
from llm.guardrails import classify_retrieval

PDF_DIR = "data/pdfs"

app = FastAPI(title="IRD Tax AI API", version="1.0.0")

@app.get("/")
def root():
    return {"message": "IRD Tax AI API is running. Visit /docs"}


# -----------------------------
# Request/Response models
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceItem(BaseModel):
    document: str
    page_range: List[int]
    section: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    status: str  # ok | missing | ambiguous


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)

import sys

def run_pipeline_rebuild():
    """
    Rebuild pipeline:
      Step 3 -> extract text
      Step 4 -> chunk
      Step 5 -> build FAISS

    Uses sys.executable to guarantee it runs inside the SAME venv.
    """
    cmds = [
        [sys.executable, "ingestion/step3_extract_pdfs.py"],
        [sys.executable, "ingestion/step4_chunker.py"],
        [sys.executable, "embedding/step5_build_faiss_local.py"],
    ]

    for c in cmds:
        p = subprocess.run(c, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Pipeline failed on: {' '.join(c)}\n{p.stderr}")


def extract_answer_from_contexts(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Step 8 (LLM) is not ready for you now, so we do a safe fallback:
    - If contexts exist, return a short snippet from best chunk.
    - This is NOT guessing; it is literally quoting context.
    """
    if not contexts:
        return "This information is not available in the provided IRD documents."

    best = contexts[0]
    text = (best.get("content") or "").strip()
    if not text:
        return "This information is not available in the provided IRD documents."

    # return first ~300 chars as a safe extractive answer
    snippet = text.replace("\n", " ")
    if len(snippet) > 300:
        snippet = snippet[:300].rstrip() + "..."
    return snippet


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDFs to data/pdfs and rebuild index.
    """
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

    # rebuild the index after upload
    try:
        run_pipeline_rebuild()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Uploaded and rebuilt index successfully",
        "files": saved
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Retrieve top chunks and apply Step 9 guardrails.
    Returns answer + sources.
    (LLM answering can be added later; for now we return safe snippet.)
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty")

    # Create retriever (loads FAISS + metadata)
    retriever = Retriever(
        index_path="vectorstore/faiss.index",
        meta_path="vectorstore/metadata.json",
        model_path="models/all-MiniLM-L6-v2",
        use_rerank=True
    )

    contexts = retriever.retrieve(q, top_k=req.top_k, fetch_k=max(12, req.top_k))

    # Step 9: missing/ambiguous handling
    status, msg = classify_retrieval(contexts)

    if status in {"missing", "ambiguous"}:
        return QueryResponse(
            answer=msg,
            sources=[],
            status=status
        )

    # Step 8 not ready => safe extractive fallback
    answer = extract_answer_from_contexts(q, contexts)

    sources = []
    for c in contexts[:req.top_k]:
        sources.append(SourceItem(
            document=c.get("document", ""),
            page_range=c.get("page_range") or [],
            section=c.get("section")
        ))

    return QueryResponse(
        answer=answer,
        sources=sources,
        status="ok"
    )

