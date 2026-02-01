import json, os, re, csv, hashlib
from typing import List, Dict, Any
from tqdm import tqdm

INPUT_JSON = "data/processed_text/ird_documents.json"
OUT_DIR = "data/chunks"
OUTPUT_JSON = os.path.join(OUT_DIR, "ird_chunks.json")
OUTPUT_CSV  = os.path.join(OUT_DIR, "ird_chunks_preview.csv")
os.makedirs(OUT_DIR, exist_ok=True)

HEADER_FOOTER_PHRASES = [
    "INLAND REVENUE DEPARTMENT",
    "GUIDE TO CORPORATE",
    "YEAR OF ASSESSMENT",
    "Commissioner General of Inland Revenue",
    "SET- E 2025/2026",
    "Asmt_CIT_003_E",
]

DIGITS_ONLY_RE = re.compile(r"^\s*\d+\s*$")
GARBAGE_LINE_RE = re.compile(r"^\s*[`•\-_]{1,5}\s*$")

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_headers_footers(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        l = line.strip()
        if not l:
            cleaned.append("")
            continue
        if DIGITS_ONLY_RE.match(l): 
            continue
        if GARBAGE_LINE_RE.match(l):
            continue
        up = l.upper()
        if any(p.upper() in up for p in HEADER_FOOTER_PHRASES):
            continue
        cleaned.append(line)
    text2 = "\n".join(cleaned)
    text2 = re.sub(r"\n{3,}", "\n\n", text2)
    return text2.strip()

def join_broken_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n")]
    out = []
    i = 0

    def is_bullet(line: str) -> bool:
        return line.startswith(("-", "•", "*", "", "(a)", "(b)")) or re.match(r"^\(?\d+(\.\d+)?[.)]\s+", line or "") is not None

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

        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt and not is_bullet(nxt) and not re.search(r"[.:;?!]\s*$", cur):
                cur = cur + " " + nxt
                i += 2
                while i < len(lines):
                    nxt2 = lines[i]
                    if (not nxt2) or is_bullet(nxt2) or re.search(r"[.:;?!]\s*$", cur):
                        break
                    cur = cur + " " + nxt2
                    i += 1
                out.append(cur)
                continue

        out.append(cur)
        i += 1

    text2 = "\n".join(out)
    text2 = re.sub(r"\n{3,}", "\n\n", text2)
    return text2.strip()

def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = remove_headers_footers(text)
    text = join_broken_lines(text)
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
    Merge consecutive pages within same (document, section) to reach larger context.
    Keeps citation by storing page_range.
    """
    # sort to be safe
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

        # merge if same doc+section AND pages are consecutive
        if doc == cur["document"] and section == cur["section"] and page == cur["end_page"] + 1:
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
    chunk_size = 520
    overlap = 80
    min_words = 120

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # ✅ merge small pages into bigger blocks first
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
            w.writerow([c["chunk_id"], c["document"], str(c["page_range"]), c["section"], c["word_len"],
                        c["content"][:400].replace("\n", " ")])

    lens = [c["word_len"] for c in chunks_out] or [0]
    print("\n--- STEP 4 STATS ---")
    print(f"Blocks after merge: {len(blocks)}")
    print(f"Chunks: {len(chunks_out)}")
    print(f"Avg words/chunk: {sum(lens)/len(lens):.1f}")
    print(f"Min words: {min(lens)}")
    print(f"Max words: {max(lens)}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Preview: {OUTPUT_CSV}")

if __name__ == "__main__":
    run()
