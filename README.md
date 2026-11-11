# RAG Knowledge Assistant

A modular retrieval-augmented generation starter with a FastAPI backend and a React + TypeScript web app. It uses Groq-hosted language models for generation, Sentence Transformers (BGE) from Hugging Face for embeddings, persisted local FAISS plus BM25 for hybrid retrieval, and PyMuPDF/python-docx for parsing.

## Features

- **Chat:** ask questions over indexed documents and see source excerpts.
- **Documents:** upload PDF, DOCX, TXT, or Markdown files; view indexed filenames/chunk counts and delete documents.
- **Hybrid retrieval:** reciprocal-rank fusion over dense FAISS and BM25 keyword results.
- **Persistence:** local FAISS index and JSON chunk metadata in `data/faiss`. Chat history stays in the active browser tab session and is not stored by the backend.
- **API:** FastAPI health/readiness endpoints and interactive API docs at `/docs`.

## Run locally

Use Node.js 22+ and Python 3.11. Create and activate a Python virtual environment, then install the backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `.env` does not exist yet, create it from the example:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set `GROK_API_KEY` in `.env` to a Groq API key, then run the backend in one terminal:

```powershell
uvicorn app.main:app --reload
```

In a second terminal, start the React frontend:

```powershell
cd C:\Users\admin\OneDrive\Desktop\aws\rag\frontend
npm install
npm run dev
```

Open the dev frontend at <http://localhost:5173>. Vite proxies `/api` requests to the local FastAPI server.

## Configuration

Configure `GROK_API_KEY` with a Groq API key, `GROK_BASE_URL` with `https://api.groq.com` (the Groq SDK adds `/openai/v1`), and set `GROK_MODEL` plus `GROK_FALLBACK_MODEL` to model IDs available to your Groq account. The backend retries the configured fallback only for model-unavailable responses. `VITE_API_BASE_URL` optionally configures the frontend API base URL; by default it uses `/api/v1`, with Vite proxying requests to FastAPI during development. `HUGGINGFACE_ENDPOINT` configures the Hugging Face Hub endpoint, and `HUGGINGFACE_MODEL_URL` selects the embedding model; the configured URL is used to derive its Hugging Face repository ID. `HF_TOKEN` is optional for public models and can authenticate access to private or gated repositories. `DATA_DIR` controls local persistence; chunk settings and retrieval count are configurable. The default vector store is FAISS. For production deployments, put the API behind an authenticated gateway, restrict network exposure, and use durable persistent storage for the FAISS data.

## API

- `GET /api/v1/health` — liveness check
- `GET /api/v1/ready` — API readiness check
- `GET /api/v1/documents` — list indexed files
- `POST /api/v1/documents/upload` — ingest one supported file (20 MB max)
- `DELETE /api/v1/documents/{document_id}` — delete its FAISS chunks
- `POST /api/v1/chat` — answer using retrieved context; optional history is supplied transiently by the client and is not stored by the API

This starter does not include user authentication or multi-tenant isolation; add those before exposing it to untrusted users. Chat history exists only in the current browser tab session and is lost when that session ends.
