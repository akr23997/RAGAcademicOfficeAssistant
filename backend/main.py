from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.rag import generate_answer


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# Create the FastAPI application ONCE
app = FastAPI(
    title="IIT Dharwad Academic Office Assistant",
    description="RAG-based academic assistance chatbot",
    version="1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)


# Request model
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


# Health check
@app.get("/api/health")
def health():
    return {"status": "running"}


# Chat endpoint
@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:
        return generate_answer(question)

    except Exception as exc:
        # Print diagnostic details in the VS Code terminal.
        print("Chat error:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail="The assistant could not process your question."
        )


# Serve the frontend
if not FRONTEND_DIR.is_dir():
    raise RuntimeError(
        f"Frontend directory not found: {FRONTEND_DIR}"
    )

app.mount(
    "/",
    StaticFiles(
        directory=str(FRONTEND_DIR),
        html=True
    ),
    name="frontend"
)