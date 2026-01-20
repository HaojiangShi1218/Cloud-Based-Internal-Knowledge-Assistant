# app/main.py
from fastapi import FastAPI
from loguru import logger
from app.config import settings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
    
app = FastAPI(title=settings.APP_NAME)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "environment": settings.ENV
    }

@app.post("/ask")
def ask_question(payload: dict):
    question = payload.get("question", "")

    if not question:
        return {"error": "Question is required"}

    # Stubbed response (RAG comes next)
    return {
        "question": question,
        "answer": "This is a placeholder answer.",
        "citations": []
    }

logger.info("Application startup complete")
