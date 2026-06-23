"""
MedInfo AI - Smart Medical Assistant with Gemini AI
Uses Gemini LLM to answer questions, analyze documents, explain drugs simply
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Load environment
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.genai as genai

# ============================================
# CONFIGURATION
# ============================================

API_KEY = os.getenv("GEMINI_API_KEY", "")
# MODEL_NAME = "gemini-2.0-flash"
print("API_KEY",API_KEY)
MODEL_NAME = "gemini-3.5-flash"

# Initialize Gemini
gemini_client = None
if API_KEY and len(API_KEY) > 10:
    gemini_client = genai.Client(api_key=API_KEY)

app = FastAPI(title="MedInfo AI", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============================================
# STORAGE FOR UPLOADED DOCUMENTS
# ============================================

uploaded_documents = {}  # doc_id -> document text
chat_history = {}  # doc_id -> [(question, answer)]

# ============================================
# GEMINI AI FUNCTIONS
# ============================================

def ask_gemini(prompt: str, system_prompt: str = "") -> str:
    """Call Gemini AI to answer questions"""
    if not gemini_client:
        return "❌ Gemini API not configured. Please add GEMINI_API_KEY to .env file."

    try:
        full_prompt = system_prompt + "\n\n" + prompt if system_prompt else prompt

        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=[full_prompt],
            config={"temperature": 0.7, "max_output_tokens": 2048}
        )

        return response.text if hasattr(response, 'text') else str(response)

    except Exception as e:
        return f"❌ Error: {str(e)}"

def explain_drug_simple(drug_name: str) -> str:
    """Explain a drug in simple terms for non-medical students"""
    prompt = f"""You are a friendly medical professor explaining medicines to students who have NO medical background.

Explain the drug "{drug_name}" in VERY SIMPLE terms:

1. WHAT IS IT? (like explaining to a 5-year-old)
2. WHAT DOES IT DO? (explain the main effect simply)
3. WHY DO PEOPLE TAKE IT? (common uses)
4. WHAT HAPPENS IF YOU TAKE IT? (effects - good and bad)
5. IMPORTANT THINGS TO KNOW (warnings, don'ts)
6. HOW DO YOU TAKE IT? (pills, injections, etc.)

Use simple words. Avoid medical jargon. If you must use a medical term, explain it immediately.

Make it educational and easy to understand for a college student who knows nothing about medicine."""

    system = """You are a medical educator who explains complex medical topics in simple, understandable language.
Always use analogies and examples. Be friendly and patient. Never make the student feel bad for not knowing."""

    return ask_gemini(prompt, system)

def answer_from_document(document_text: str, question: str) -> str:
    """Answer a question based on an uploaded document"""
    prompt = f"""Based on the following clinical document, answer the user's question.

DOCUMENT:
{document_text}

QUESTION: {question}

Answer based ONLY on the document. If the document doesn't contain enough information to answer, say so clearly."""

    system = """You are a medical AI assistant analyzing patient documents.
Provide accurate, helpful answers based ONLY on the document content.
If unsure, acknowledge the limitation."""

    return ask_gemini(prompt, system)

def suggest_based_on_document(document_text: str, user_context: str = "") -> str:
    """Give suggestions based on the document"""
    prompt = f"""Based on this clinical patient document, provide helpful medical information and suggestions.

DOCUMENT:
{document_text}

User's context: {user_context if user_context else "No additional context provided"}

Provide:
1. POSSIBLE CONDITIONS - What conditions might be indicated in this document?
2. RECOMMENDED TESTS - What tests might be needed?
3. MEDICATIONS TO KNOW ABOUT - What medications are mentioned or might be relevant?
4. LIFESTYLE CHANGES - What lifestyle modifications might help?
5. WHEN TO SEE A DOCTOR - What symptoms should prompt immediate medical attention?

IMPORTANT: Always remind the user that this is NOT medical advice and they should consult a healthcare professional."""

    system = """You are a helpful medical information assistant.
Provide educational information only. Never claim to diagnose. Always include disclaimers."""

    return ask_gemini(prompt, system)

def explain_medical_term(term: str) -> str:
    """Explain a medical term in simple language"""
    prompt = f"""Explain this medical term/concept: "{term}"

Explain it like you're teaching a college student who has NO medical knowledge. Use:
- Simple everyday analogies
- Real-world examples
- Avoid technical jargon OR immediately explain any terms you use
- Keep it short but informative"""

    system = "You are a medical educator. Make complex topics simple and accessible."

    return ask_gemini(prompt, system)

def compare_medications(meds: List[str]) -> str:
    """Compare multiple medications in simple terms"""
    meds_list = ", ".join(meds)
    prompt = f"""Compare these medications: {meds_list}

For EACH medication, provide in simple terms:
1. What is it?
2. What does it treat?
3. Key side effects (in plain language)
4. How is it taken?
5. Important warnings

Then compare them:
- Which is better for what?
- Can they be taken together?
- What should you not mix?

Make it easy to understand for someone with NO medical background."""

    system = "You are a medical educator. Compare medications clearly and simply."

    return ask_gemini(prompt, system)

# ============================================
# REQUEST MODELS
# ============================================

class AskQuestion(BaseModel):
    question: str

class UploadText(BaseModel):
    text: str
    title: Optional[str] = "Untitled Document"

class ChatWithDoc(BaseModel):
    doc_id: str
    question: str

class GetSuggestions(BaseModel):
    doc_id: str
    context: Optional[str] = ""

class ExplainTerm(BaseModel):
    term: str

class CompareMeds(BaseModel):
    medicines: List[str]

class GetDrugInfo(BaseModel):
    drug: str

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "MedInfo AI is running!",
        "gemini_ready": gemini_client is not None
    }

# ------------------------------------------
# 1. UPLOAD DOCUMENT
# ------------------------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a clinical document"""
    content = await file.read()
    text = content.decode("utf-8")

    doc_id = f"doc_{len(uploaded_documents) + 1}"
    uploaded_documents[doc_id] = text
    chat_history[doc_id] = []

    return {
        "status": "success",
        "doc_id": doc_id,
        "message": f"Document '{file.filename}' uploaded successfully!",
        "preview": text[:500] + "..." if len(text) > 500 else text
    }

@app.post("/upload_text")
def upload_text(req: UploadText):
    """Upload text directly"""
    doc_id = f"doc_{len(uploaded_documents) + 1}"
    uploaded_documents[doc_id] = req.text
    chat_history[doc_id] = []

    return {
        "status": "success",
        "doc_id": doc_id,
        "message": "Document uploaded successfully!"
    }

# ------------------------------------------
# 2. CHAT WITH DOCUMENT (RAG)
# ------------------------------------------
@app.post("/chat")
def chat_with_document(req: ChatWithDoc):
    """Ask questions about an uploaded document"""
    if req.doc_id not in uploaded_documents:
        return {"status": "error", "error": "Document not found"}

    doc_text = uploaded_documents[req.doc_id]
    answer = answer_from_document(doc_text, req.question)

    # Save to history
    chat_history[req.doc_id].append({
        "question": req.question,
        "answer": answer
    })

    return {
        "status": "success",
        "answer": answer,
        "disclaimer": "⚠️ This is based on the uploaded document. Consult healthcare professionals for medical advice."
    }

# ------------------------------------------
# 3. GET SUGGESTIONS FROM DOCUMENT
# ------------------------------------------
@app.post("/suggest")
def get_suggestions(req: GetSuggestions):
    """Get suggestions based on document"""
    if req.doc_id not in uploaded_documents:
        return {"status": "error", "error": "Document not found"}

    doc_text = uploaded_documents[req.doc_id]
    suggestions = suggest_based_on_document(doc_text, req.context)

    return {
        "status": "success",
        "suggestions": suggestions,
        "disclaimer": "⚠️ Educational purposes only. Consult healthcare professionals."
    }

# ------------------------------------------
# 4. EXPLAIN DRUG SIMPLY (FOR NON-MEDICAL STUDENTS)
# ------------------------------------------
@app.post("/explain_drug")
def explain_drug(req: GetDrugInfo):
    """Explain a drug in simple terms"""
    explanation = explain_drug_simple(req.drug)

    return {
        "status": "success",
        "drug": req.drug,
        "explanation": explanation,
        "disclaimer": "⚠️ Educational only. Consult doctor/pharmacist for medical advice."
    }

# ------------------------------------------
# 5. EXPLAIN MEDICAL TERM
# ------------------------------------------
@app.post("/explain_term")
def explain_term(req: ExplainTerm):
    """Explain a medical term simply"""
    explanation = explain_medical_term(req.term)

    return {
        "status": "success",
        "term": req.term,
        "explanation": explanation
    }

# ------------------------------------------
# 6. COMPARE MEDICATIONS
# ------------------------------------------
@app.post("/compare")
def compare_meds(req: CompareMeds):
    """Compare multiple medications"""
    comparison = compare_medications(req.medicines)

    return {
        "status": "success",
        "comparison": comparison,
        "disclaimer": "⚠️ Consult pharmacist for complete drug interaction information."
    }

# ------------------------------------------
# 7. GENERAL Q&A WITH GEMINI
# ------------------------------------------
@app.post("/ask")
def ask_anything(req: AskQuestion):
    """Ask any medical question - Gemini will answer intelligently"""
    prompt = f"""You are a helpful medical information assistant.

Question: {req.question}

Provide a clear, accurate, educational answer. If you're not sure about something, say so.
Always remind users to consult healthcare professionals for medical advice."""

    system = """You are a medical AI assistant. Provide accurate, helpful medical information.
Use simple language. Include disclaimers about consulting healthcare professionals.
Never provide specific medical diagnoses."""

    answer = ask_gemini(prompt, system)

    return {
        "status": "success",
        "answer": answer,
        "disclaimer": "⚠️ This is educational information only. Always consult healthcare professionals."
    }

# ------------------------------------------
# 8. LIST DOCUMENTS
# ------------------------------------------
@app.get("/documents")
def list_documents():
    """List all uploaded documents"""
    return {
        "documents": [
            {"doc_id": doc_id, "preview": text[:100] + "..."}
            for doc_id, text in uploaded_documents.items()
        ]
    }

# ------------------------------------------
# 9. GET CHAT HISTORY
# ------------------------------------------
@app.get("/chat/{doc_id}")
def get_chat_history(doc_id: str):
    """Get chat history for a document"""
    if doc_id not in chat_history:
        return {"status": "error", "error": "Document not found"}

    return {
        "status": "success",
        "history": chat_history[doc_id]
    }

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🏥 MedInfo AI - Smart Medical Assistant")
    print("=" * 60)
    print(f"Gemini API: {'✅ Ready' if gemini_client else '❌ Not configured'}")
    print("API: http://localhost:8001")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8002)