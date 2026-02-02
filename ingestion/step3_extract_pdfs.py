import os
import json
import pdfplumber
import re

PDF_DIR = "data/pdfs"
OUT_JSON = "data/processed_text/ird_documents.json"
os.makedirs("data/processed_text", exist_ok=True)


def detect_section(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Try to find a strong heading like PN/IT/2025-01 first
    for l in lines[:10]:
        if re.match(r"^PN/IT/\d{4}-\d{2}$", l):
            return l

    # Otherwise just keep it stable
    return "Not specified"


def main():
    if not os.path.exists(PDF_DIR):
        raise FileNotFoundError(f"PDF folder not found: {PDF_DIR}")

    records = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    for pdf_name in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_name)

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                section = detect_section(text)

                records.append({
                    "document": pdf_name,
                    "page": i,
                    "section": section,
                    "content": text
                })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Step 3 done. Pages extracted: {len(records)}")
    print(f"Saved to: {OUT_JSON}")

if __name__ == "__main__":
    main()
