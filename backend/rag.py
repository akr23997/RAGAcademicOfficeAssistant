import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from backend.ingest import extract_pdf_text, chunk_documents


# --------------------------------------------------
# 1. Project paths and environment configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db"

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        f"Set GROQ_API_KEY in your .env file: {ENV_FILE}"
    )


# --------------------------------------------------
# 2. Groq client
# --------------------------------------------------

llm_client = Groq(api_key=GROQ_API_KEY)


# --------------------------------------------------
# 3. Embedding model and ChromaDB
# --------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

DATABASE_DIR.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=str(DATABASE_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="iitd_academic_documents"
)


# --------------------------------------------------
# 4. Build the institutional knowledge base
# --------------------------------------------------

def build_knowledge_base():
    """Extract, chunk, embed, and index institutional PDFs."""

    if not DOCUMENTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR}"
        )

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    pages = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        pages.extend(extract_pdf_text(pdf_path))

    if not pages:
        raise ValueError(
            "No extractable PDF text found. "
            "Scanned documents may require OCR."
        )

    chunks = chunk_documents(pages)

    if not chunks:
        raise ValueError("No document chunks were generated.")

    # Remove the previous chunks indexed by this function.
    existing = collection.get(
        where={"batch": "current"},
        include=[]
    )

    existing_ids = existing.get("ids", [])

    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"Removed {len(existing_ids)} previous chunks.")

    ids = [f"current_{i}" for i in range(len(chunks))]
    texts = [item["text"] for item in chunks]

    metadatas = [
        {
            "source": item["source"],
            "page": int(item["page"]),
            "batch": "current"
        }
        for item in chunks
    ]

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        embeddings = embedding_model.encode(
            texts[start:end],
            normalize_embeddings=True
        ).tolist()

        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings,
            metadatas=metadatas[start:end]
        )

        print(f"Indexed chunks {start + 1}-{end}")

    print(f"Successfully indexed {len(chunks)} chunks.")
    print(f"Total chunks in database: {collection.count()}")


# --------------------------------------------------
# 5. Keyword detection
# --------------------------------------------------

def detect_search_terms(question):
    """Identify document sections likely to answer the question."""

    q = question.lower()

    terms = []

    if (
        "peps" in q
        and any(word in q for word in [
            "elective", "electives", "course", "courses", "list"
        ])
    ):
        terms.extend([
            "list of electives",
            "list of electives for peps",
            "power system dynamics and control",
            "design of magnetics for power electronics"
        ])

    if any(word in q for word in [
        "holiday", "holidays", "public holiday", "restricted holiday"
    ]):
        terms.extend([
            "list of holidays during the calendar year",
            "list of restricted holidays",
            "holiday date day"
        ])

    # Remove duplicate search terms.
    return list(dict.fromkeys(terms))


# --------------------------------------------------
# 6. Retrieve relevant document passages
# --------------------------------------------------

def retrieve_documents(question, top_k=10):
    """Retrieve relevant passages using vector and keyword search."""

    retrieved = {}

    def add_passage(text, metadata):
        if not text or not metadata:
            return

        key = (
            metadata.get("source", ""),
            metadata.get("page", 0),
            text
        )

        retrieved[key] = {
            "text": text,
            "source": metadata.get("source", "Unknown document"),
            "page": metadata.get("page", 0)
        }

    # A. Semantic retrieval through ChromaDB.
    total = collection.count()

    if total:
        query_embedding = embedding_model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"]
        )

        for text, metadata in zip(
            results["documents"][0],
            results["metadatas"][0]
        ):
            add_passage(text, metadata)

    # B. Keyword-based retrieval from indexed chunks.
    # This helps find lists whose headings may not rank highly
    # in semantic similarity search.
    search_terms = detect_search_terms(question)

    if search_terms and total:
        all_chunks = collection.get(
            include=["documents", "metadatas"]
        )

        for text, metadata in zip(
            all_chunks["documents"],
            all_chunks["metadatas"]
        ):
            searchable = text.lower()

            if any(term in searchable for term in search_terms):
                add_passage(text, metadata)

    passages = list(retrieved.values())

    print(f"Question: {question}")
    print(f"Retrieved passages: {len(passages)}")

    for passage in passages:
        print(
            f"  Source: {passage['source']}, "
            f"page {passage['page']}"
        )

    return passages


# --------------------------------------------------
# 7. Generate answers using Groq
# --------------------------------------------------

def generate_answer(question):
    """Generate answers grounded in retrieved institutional documents."""

    passages = retrieve_documents(question, top_k=10)

    if not passages:
        return {
            "answer": (
                "I could not find supporting information in the "
                "available IIT Dharwad documents. Please consult "
                "the relevant academic office."
            ),
            "sources": []
        }

    context_parts = []
    sources = []

    for passage in passages:
        context_parts.append(
            f"Document: {passage['source']}\n"
            f"Page: {passage['page']}\n"
            f"Content:\n{passage['text']}"
        )

        source = {
            "document": passage["source"],
            "page": passage["page"]
        }

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    prompt = f"""
You are the IIT Dharwad Academic Office Assistant.

Answer the student's question using only the document excerpts below.

Instructions:
- For questions asking for lists, include every relevant item
  available in the supplied excerpts.
- Preserve the names, dates, and wording of courses and holidays.
- Distinguish regular holidays from restricted holidays.
- Distinguish proposed electives from other electives.
- Do not invent information or dates.
- If the excerpts do not contain the requested information,
  say that it could not be verified from the retrieved documents.
- Do not confuse programme credit requirements with course lists.
- Provide a concise, direct answer.

Document excerpts:
{context}

Student question:
{question}
"""

    try:
        response = llm_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the IIT Dharwad Academic Office Assistant. "
                        "Answer from the supplied institutional documents. "
                        "Do not invent official information. Return a "
                        "concise final answer."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.2
        )

        if not response.choices:
            answer = ""
        else:
            choice = response.choices[0]
            answer = choice.message.content or ""

            print("Groq finish reason:", choice.finish_reason)

        if not answer.strip():
            answer = (
                "Groq did not return a final answer. "
                "Please try again."
            )

        return {
            "answer": answer.strip(),
            "sources": sources
        }

    except Exception as exc:
        print("Groq inference error:", repr(exc))

        return {
            "answer": (
                "I retrieved institutional documents, but Groq "
                "could not generate a response. Please check "
                "the backend terminal."
            ),
            "sources": sources
        }