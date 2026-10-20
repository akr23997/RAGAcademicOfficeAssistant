from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"


def extract_pdf_text(pdf_path):
    """Extract text from every page of a PDF."""

    pdf_path = Path(pdf_path)
    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "text": text,
                    "page": page_number,
                    "source": pdf_path.name
                })

    print(
        f"Extracted {len(pages)} text pages "
        f"from {pdf_path.name}"
    )

    return pages


def chunk_documents(pages):
    """Divide document pages into smaller text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks


if __name__ == "__main__":
    all_pages = []

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    print("Documents folder:", DOCUMENTS_DIR)
    print("PDF files found:", len(pdf_files))

    for pdf_path in pdf_files:
        all_pages.extend(extract_pdf_text(pdf_path))

    all_chunks = chunk_documents(all_pages)

    print("\nTotal pages extracted:", len(all_pages))
    print("Total chunks created:", len(all_chunks))

    if all_chunks:
        print("\nFirst chunk:")
        print(all_chunks[0]["text"])
    else:
        print(
            "\nNo chunks were created. "
            "Check whether the PDFs contain extractable text."
        )