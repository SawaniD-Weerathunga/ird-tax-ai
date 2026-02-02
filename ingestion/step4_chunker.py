import json, os, re, csv, hashlib
from typing import List, Dict, Any
from tqdm import tqdm

INPUT_JSON = "data/processed_text/ird_documents.json"
OUT_DIR = "data/chunks"
OUTPUT_JSON = os.path.join(OUT_DIR, "ird_chunks.json")
OUTPUT_CSV  = os.path.join(OUT_DIR, "ird_chunks_preview.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ✅ NEW: limit merged blocks to N pages (better page_range citations)
MAX_PAGES_PER_BLOCK = 3   # try 3 (best balance). use 2 for tighter citations.

# Phrases that often appear in page headers/footers
HEADER_FOOTER_PHRASES = [
    "INLAND REVENUE DEPARTMENT",
    "GUIDE TO CORPORATE",
    "Commissioner General of Inland Revenue",
    "SET- E 2025/2026",
    "Asmt_CIT_003_E",
]

DIGITS_ONLY_RE = re.compile(r"^\s*\d+\s*$")
# lines that are only symbols like `, •, ----, ___ etc.
SYMBOL_ONLY_RE = re.compile(r"^\s*[\W_]{1,20}\s*$")

# Common header patterns that vary per document
YEAR_OF_ASSESSMENT_LINE_RE = re.compile(r"(?i)\byear\s+of\s+assessment\b.*")
GUIDE_CIT_HEADER_RE = re.compile(r"(?i)\bguide\s+to\s+corporate\b.*\breturn\s+of\s+income\s+tax\b.*")

# sometimes PDFs repeat "GUIDE TO CORPORATE RETURN..." lines across pages
REPEATED_GUIDE_LINE_RE = re.compile(r"(?i)^\s*guide\s+to\s+corporate.*$")

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_headers_footers(text: str) -> str:
    """
    Remove obvious headers/footers and noisy standalone lines.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        l = line.strip()

        # keep blank lines (paragraph breaks)
        if not l:
            cleaned.append("")
            continue

        # remove page numbers
        if DIGITS_ONLY_RE.match(l):
            continue

        # remove lines that are only symbols (e.g., `, •, ----)
        if SYMBOL_ONLY_RE.match(l):
            continue

        up = l.upper()

        # remove known header/footer phrases
        if any(p.upper() in up for p in HEADER_FOOTER_PHRASES):
            continue

        # remove "YEAR OF ASSESSMENT ..." header lines (more robust)
        if YEAR_OF_ASSESSMENT_LINE_RE.match(l):
            continue

        # remove repeated "guide to corporate..." headers (varies)
        if GUIDE_CIT_HEADER_RE.match(l):
            continue

        # remove repeated guide line variants (safe)
        if REPEATED_GUIDE_LINE_RE.match(l) and len(l.split()) < 18:
            continue

        cleaned.append(line)

    text2 = "\n".join(cleaned)
    text2 = re.sub(r"\n{3,}", "\n\n", text2)
    return text2.strip()

def join_broken_lines(text: str) -> str:
    """
    Joins lines that were broken by PDF layout.
    Keeps bullets/numbered points as separate lines.
    """
    raw_lines = text.split("\n")
    lines = [ln.strip() for ln in raw_lines]

    def is_bullet(line: str) -> bool:
        if not line:
            return False
        # common bullets seen in IRD PDFs
        if line.startswith(("-", "•", "*", "")):
            return True
        # (a) (b) style
        if re.match(r"^\([a-zA-Z]\)\s+", line):
            return True
        # 1. 2. 3) 1.1 etc
        if re.match(r"^\d+(\.\d+)*[.)]\s+", line):
            return True
        return False

    out = []
    i = 0
    while i < len(lines):
        cur = lines[i]

        if not cur:
            out.append("")
            i += 1
            continue

        if is_bullet(cur):
            out.append(cur)
            i += 1
            continue

        # Try to join with next lines if the current doesn't look like a sentence end.
        combined = cur
        while i + 1 < len(lines):
            nxt = lines[i + 1]

            if not nxt:
                break
            if is_bullet(nxt):
                break

            # If current already ends like a sentence/heading, stop.
            if re.search(r"[.:;?!]\s*$", combined):
                break

            # If current line looks like a heading (short) and next is also short, don't over-join.
            if len(combined.split()) <= 4 and len(nxt.split()) <= 4:
                break

            combined = combined + " " + nxt
            i += 1

            if re.search(r"[.:;?!]\s*$", combined):
                break

        out.append(combined)
        i += 1

    text2 = "\n".join(out)
    text2 = re.sub(r"\n{3,}", "\n\n", text2)
    return text2.strip()

def remove_inline_noise(text: str) -> str:
    """
    Remove inline patterns that slipped through line-based header removal.
    """
    text = re.sub(r"(?i)\byear\s+of\s+assessment\b\s*[-–]?\s*\d{4}\s*/\s*\d{4}", " ", text)
    text = re.sub(r"(?i)\byear\s+of\s+assessment\b", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = remove_headers_footers(text)
    text = join_broken_lines(text)
    text = remove_inline_noise(text)
    text = normalize_whitespace(text)
    return text

def chunk_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]).strip())
        if end == len(words):
            break
        start += step
    return chunks

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_chunk_id(doc: str, start_page: int, end_page: int, idx: int) -> str:
    safe_doc = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc)
    return f"{safe_doc}_p{start_page}-{end_page}_c{idx}"

def merge_pages(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive pages within same (document, section),
    but limit each merged block to MAX_PAGES_PER_BLOCK pages.
    """
    records = sorted(records, key=lambda r: (r.get("document",""), int(r.get("page",0))))
    merged = []
    cur = None

    for r in records:
        doc = r.get("document","unknown.pdf")
        page = int(r.get("page",0))
        section = r.get("section","Not specified")
        content = clean_text(r.get("content",""))

        if not content:
            continue

        if cur is None:
            cur = {
                "document": doc,
                "section": section,
                "start_page": page,
                "end_page": page,
                "content": content
            }
            continue

        can_merge = (
            doc == cur["document"]
            and section == cur["section"]
            and page == cur["end_page"] + 1
        )

        cur_page_count = (cur["end_page"] - cur["start_page"] + 1)

        if can_merge and cur_page_count < MAX_PAGES_PER_BLOCK:
            cur["content"] += "\n\n" + content
            cur["end_page"] = page
        else:
            merged.append(cur)
            cur = {
                "document": doc,
                "section": section,
                "start_page": page,
                "end_page": page,
                "content": content
            }

    if cur is not None:
        merged.append(cur)

    return merged

def run():
    chunk_size = 380
    overlap = 60
    min_words = 120

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        pages = json.load(f)

    blocks = merge_pages(pages)

    chunks_out = []
    seen = set()

    for b in tqdm(blocks, desc="Merge+Chunk"):
        doc = b["document"]
        section = b["section"]
        sp = b["start_page"]
        ep = b["end_page"]
        text = b["content"]

        for idx, ch in enumerate(chunk_words(text, chunk_size, overlap), start=1):
            if len(ch.split()) < min_words:
                continue
            h = stable_hash(ch)
            if h in seen:
                continue
            seen.add(h)

            chunks_out.append({
                "chunk_id": make_chunk_id(doc, sp, ep, idx),
                "document": doc,
                "page_range": [sp, ep],
                "section": section,
                "content": ch,
                "word_len": len(ch.split()),
                "char_len": len(ch)
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks_out, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "document", "page_range", "section", "word_len", "preview"])
        for c in chunks_out:
            w.writerow([
                c["chunk_id"],
                c["document"],
                str(c["page_range"]),
                c["section"],
                c["word_len"],
                c["content"][:400].replace("\n", " ")
            ])

    lens = [c["word_len"] for c in chunks_out] or [0]
    print("\n--- STEP 4 STATS ---")
    print(f"Blocks after merge: {len(blocks)} (MAX_PAGES_PER_BLOCK={MAX_PAGES_PER_BLOCK})")
    print(f"Chunks: {len(chunks_out)}")
    print(f"Avg words/chunk: {sum(lens)/len(lens):.1f}")
    print(f"Min words: {min(lens)}")
    print(f"Max words: {max(lens)}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Preview: {OUTPUT_CSV}")

if __name__ == "__main__":
    run()
