import os
import re
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from embedding.retriever import Retriever
from llm.guardrails import classify_retrieval
from llm.prompt_builder import build_system_prompt, build_user_prompt
from llm.disclaimer import append_disclaimer

PDF_DIR = "data/pdfs"

# ---- Ollama settings (can change without code edits)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # change to what you have pulled
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
    """
    Rebuild pipeline:
      Step 3 -> extract text
      Step 4 -> chunk
      Step 5 -> build FAISS
    """
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


YEAR_PATTERN = re.compile(r"\b(20\d{2}\s*/\s*20\d{2})\b")
NUM_TOKEN_RE = re.compile(r"\b\d[\d,]*\b%?")
NON_DOMAIN_GEO_TERMS = {
    "europe", "eu", "european union", "uk", "united kingdom", "england", "scotland",
    "wales", "ireland", "usa", "united states", "america", "canada", "australia",
    "india", "singapore", "malaysia", "germany", "france", "italy", "spain",
    "netherlands", "sweden", "norway", "denmark", "finland", "japan", "china"
}
SL_DOMAIN_TERMS = {
    "sri lanka", "ird", "inland revenue", "inland revenue department",
    "cit", "pit", "set", "income tax", "withholding", "apit", "payE".lower(),
    "year of assessment", "yoa"
}


def extract_years(text: str) -> List[str]:
    if not text:
        return []
    return [y.replace(" ", "") for y in YEAR_PATTERN.findall(text)]


def contexts_contain_year(contexts: List[Dict[str, Any]], year: str) -> bool:
    if not year:
        return False
    y = year.replace(" ", "")
    for c in contexts:
        content = (c.get("content") or "")
        if y in content.replace(" ", ""):
            return True
    return False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def query_mentions_non_domain_geo(question: str) -> bool:
    q = _normalize_text(question)
    if not q:
        return False
    # If the user explicitly mentions Sri Lanka/IRD, treat as in-domain
    if any(term in q for term in SL_DOMAIN_TERMS):
        return False
    return any(term in q for term in NON_DOMAIN_GEO_TERMS)


def has_min_keyword_overlap(question: str, contexts: List[Dict[str, Any]], min_overlap: float = 0.25) -> bool:
    """
    Basic lexical guardrail: require some overlap of non-trivial tokens
    between question and at least one context.
    """
    q = _normalize_text(question)
    if not q:
        return False
    q_tokens = {t for t in re.split(r"[^a-z0-9/]+", q) if t and len(t) > 2}
    if not q_tokens:
        return False

    for c in contexts:
        text = _normalize_text(c.get("content") or "")
        if not text:
            continue
        t_tokens = {t for t in re.split(r"[^a-z0-9/]+", text) if t and len(t) > 2}
        if not t_tokens:
            continue
        overlap = len(q_tokens & t_tokens) / len(q_tokens)
        if overlap >= min_overlap:
            return True
    return False


def has_citations_block(answer: str) -> bool:
    if not answer:
        return False
    return "Citations:" in answer


def normalize_doc_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def parse_citations(answer: str) -> List[str]:
    """
    Extract document names from a Citations block.
    Expected format:
    Citations:
    - <document>, pages <x-y>, section <section>
    """
    if not answer or "Citations:" not in answer:
        return []
    lines = answer.splitlines()
    docs = []
    in_block = False
    for ln in lines:
        if ln.strip().lower().startswith("citations:"):
            in_block = True
            continue
        if in_block:
            if not ln.strip():
                continue
            if not ln.strip().startswith("-"):
                # end of block if formatting changes
                break
            # take text after "- " up to first comma as document name
            item = ln.strip()[1:].strip()
            doc = item.split(",")[0].strip()
            if doc:
                docs.append(normalize_doc_name(doc))
    return docs


def validate_citations(answer: str, sources: List[SourceItem]) -> bool:
    cited = parse_citations(answer)
    if not cited:
        return False
    source_docs = {normalize_doc_name(s.document) for s in sources if s.document}
    # all cited docs must exist in sources
    return all(c in source_docs for c in cited)


def answer_numbers_supported(answer: str, contexts: List[Dict[str, Any]]) -> bool:
    """
    Basic hallucination guard: all numeric tokens in answer
    must appear somewhere in the retrieved contexts.
    """
    if not answer:
        return False
    all_ctx = " ".join((c.get("content") or "") for c in contexts).replace(" ", "")
    nums = {t.replace(" ", "") for t in NUM_TOKEN_RE.findall(answer)}
    if not nums:
        return True
    for n in nums:
        if n.replace(",", "") not in all_ctx.replace(",", ""):
            return False
    return True


def format_citations(sources: List[SourceItem], limit: int = 3) -> str:
    items = []
    for s in sources[:limit]:
        if not s.document or not s.page_range:
            continue
        pr = s.page_range
        if len(pr) == 1:
            pages = f"{pr[0]}"
        else:
            pages = f"{pr[0]}-{pr[-1]}"
        section = s.section or "Not specified"
        items.append(f"- {s.document}, pages {pages}, section {section}")
    if not items:
        return ""
    return "Citations:\n" + "\n".join(items)


def maybe_attach_citations(answer: str, sources: List[SourceItem]) -> str:
    if has_citations_block(answer):
        return answer
    cits = format_citations(sources)
    if not cits:
        return answer
    return answer.rstrip() + "\n\n" + cits


def is_ambiguous_query(question: str) -> bool:
    q = _normalize_text(question)
    if not q:
        return False
    # generic tax rate/relief questions without tax type are ambiguous
    generic = any(k in q for k in ["rate", "rates", "relief", "threshold", "slab", "exemption"])
    has_tax_type = any(k in q for k in ["cit", "corporate", "pit", "personal", "set", "withholding", "apit"])
    has_year = bool(extract_years(q))
    if generic and not has_tax_type:
        # if no year too, treat as ambiguous
        if not has_year:
            return True
    return False


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

    try:
        run_pipeline_rebuild()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Uploaded and rebuilt index successfully", "files": saved}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Step 6: Retrieve
    Step 9: Guardrails (missing/ambiguous)
    Step 8: LLM (optional) with fallback
    Step 11: Disclaimer appended to EVERY answer/message
    """
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
    # If query is specific (tax type/year), don't auto-mark ambiguous on close scores
    if status == "ambiguous" and not is_ambiguous_query(q):
        status, msg = "ok", ""

    # Build sources (even if missing/ambiguous, keep empty as per your rule)
    sources = [
        SourceItem(
            document=c.get("document", ""),
            page_range=c.get("page_range") or [],
            section=c.get("section"),
        )
        for c in contexts[: req.top_k]
    ]

    # If missing/ambiguous -> return message (WITH disclaimer)
    if status in {"missing", "ambiguous"}:
        return QueryResponse(
            answer=append_disclaimer(msg),
            sources=[],
            status=status,
            used_llm=False,
        )

    # Year-specific guardrail: if question includes a Y/A but contexts don't, treat as missing
    years_in_q = extract_years(q)
    if years_in_q:
        # If any year mentioned isn't supported by retrieved contexts, return missing
        if not any(contexts_contain_year(contexts, y) for y in years_in_q):
            return QueryResponse(
                answer=append_disclaimer(
                    "This information is not available for the requested year in the provided IRD documents."
                ),
                sources=[],
                status="missing",
                used_llm=False,
        )

    # Ambiguity guardrail based on missing tax type/year
    if is_ambiguous_query(q):
        return QueryResponse(
            answer=append_disclaimer(
                "Your question may be ambiguous. Please specify the tax type (PIT/CIT/SET) and year of assessment."
            ),
            sources=[],
            status="ambiguous",
            used_llm=False,
        )

    # Out-of-domain geo guardrail (e.g., Europe VAT)
    if query_mentions_non_domain_geo(q):
        return QueryResponse(
            answer=append_disclaimer(
                "This question appears to be outside the scope of Sri Lanka IRD documents."
            ),
            sources=[],
            status="missing",
            used_llm=False,
        )

    # Keyword overlap guardrail to reduce false positives from weak retrieval
    if not has_min_keyword_overlap(q, contexts):
        return QueryResponse(
            answer=append_disclaimer(
                "This information is not available in the provided IRD documents."
            ),
            sources=[],
            status="missing",
            used_llm=False,
        )

    # Step 8: LLM answer (optional)
    if req.use_llm:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(q, contexts)

        try:
            answer = call_ollama_chat(system_prompt, user_prompt)
            if answer and has_citations_block(answer):
                if not validate_citations(answer, sources):
                    raise ValueError("LLM citations do not match retrieved sources")
                if not answer_numbers_supported(answer, contexts):
                    raise ValueError("LLM answer contains unsupported numeric claims")
                return QueryResponse(
                    answer=append_disclaimer(answer),
                    sources=sources,
                    status="ok",
                    used_llm=True,
                )
        except Exception:
            # Any Ollama/model error => fallback safely
            pass

    # fallback: extractive answer (WITH disclaimer)
    answer = safe_extractive_answer(contexts)
    answer = maybe_attach_citations(answer, sources)
    return QueryResponse(
        answer=append_disclaimer(answer),
        sources=sources,
        status="ok",
        used_llm=False,
    )
