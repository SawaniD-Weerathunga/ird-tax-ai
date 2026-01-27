import fitz  # PyMuPDF
import json
import os

RAW_PDF_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/processed_text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_section(text):
    lines = text.split("\n")

    for line in lines[:5]:  # check only top 5 lines
        clean_line = line.strip()

        if len(clean_line) < 5:
            continue

        # Rule 1: ALL CAPS headings
        if clean_line.isupper() and len(clean_line) < 80:
            return clean_line.title()

        # Rule 2: Numbered headings (e.g., 5.1, 3.)
        if clean_line[0].isdigit() and "." in clean_line:
            return clean_line

        # Rule 3: Keyword-based detection
        keywords = ["RATE", "TAX", "EXEMPTION", "PENALTY", "COMPUTATION"]
        for kw in keywords:
            if kw in clean_line.upper():
                return clean_line

    return "Not specified"


def extract_text_from_pdf(pdf_path):
    document_name = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)

    extracted_pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")

        # Basic cleaning
        text = text.replace("\n\n", "\n").strip()

        if not text:
            continue
        
        section = detect_section(text)

        page_data = {
            "document": document_name,
            "page": page_index + 1,  # Pages start from 1
            "section": section,
            "content": text
        }

        extracted_pages.append(page_data)

    return extracted_pages


def process_all_pdfs():
    all_documents = []

    for file in os.listdir(RAW_PDF_DIR):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(RAW_PDF_DIR, file)
            print(f"Processing: {file}")

            pages = extract_text_from_pdf(pdf_path)
            all_documents.extend(pages)

    output_file = os.path.join(OUTPUT_DIR, "ird_documents.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)

    print(f"\nExtraction complete!")
    print(f"Saved to: {output_file}")
    print(f"Total pages extracted: {len(all_documents)}")


if __name__ == "__main__":
    process_all_pdfs()
