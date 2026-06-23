# MedInfo AI - Smart Medical Assistant

A simple, AI-powered medical information assistant for students. Uses Google Gemini AI to explain medical concepts in plain language.

## Features

| Feature | Description |
|---------|-------------|
|  **Upload Documents** | Upload clinical patient documents |
|  **Chat with Documents** | Ask questions about uploaded documents |
|  **Explain Drugs** | Get simple explanations of any medication |
|  **Medical Terms** | Understand medical jargon in simple words |
|  **Compare Meds** | Compare multiple medications side-by-side |
|  **Ask Anything** | Ask any medical question to AI |

## Quick Start

### 1. Install Requirements
```bash
pip install fastapi uvicorn google-genai python-dotenv pydantic
```

### 2. Get Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Add to `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the Server
```bash
python3 medinfo_ai.py
```

### 4. Open the UI
Open `index.html` in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/upload_text` | POST | Upload document text |
| `/documents` | GET | List uploaded documents |
| `/chat` | POST | Chat with a document |
| `/explain_drug` | POST | Explain a drug simply |
| `/explain_term` | POST | Explain medical term |
| `/compare` | POST | Compare medications |
| `/ask` | POST | Ask any medical question |

## Example Requests

### Explain a Drug
```bash
curl -X POST http://localhost:8001/explain_drug \
  -H "Content-Type: application/json" \
  -d '{"drug": "metformin"}'
```

### Compare Medications
```bash
curl -X POST http://localhost:8001/compare \
  -H "Content-Type: application/json" \
  -d '{"medicines": ["metformin", "lisinopril"]}'
```

### Ask a Question
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is diabetes?"}'
```

## For Non-Medical Students

This tool is designed specifically for students with NO medical background:

- **Simple Language**: No complex medical jargon
- **Analogies**: Uses everyday examples to explain
- **Educational**: Learn about drugs and diseases easily
- **Disclaimer**: Always reminds to consult healthcare professionals

## Important Notes

- This is for **educational purposes only**
- Always consult a healthcare professional for medical advice
- The AI provides information, not diagnoses
- Check your API quota in Google AI Studio

## 📁 Project Files

```
medinfo_ai.py    - Backend API (FastAPI + Gemini)
index.html       - Frontend UI
README.md        - This file
.env             - Your API key (create this)
```

## 🔧 Requirements

- Python 3.8+
- Google Gemini API Key
