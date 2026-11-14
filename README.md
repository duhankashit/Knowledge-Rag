# RAG Knowledge Assistant

A modular retrieval-augmented generation starter with a FastAPI backend and a React + TypeScript web app. It uses Groq-hosted language models for generation, Sentence Transformers (BGE) from Hugging Face for embeddings, persisted local FAISS plus BM25 for hybrid retrieval, and PyMuPDF/python-docx for parsing.

## Features

- **Chat:** ask questions over indexed documents and see source excerpts.
- **Evaluation:** run RAGAS or DeepEval on demand from an assistant answer to review retrieval, generation, safety, and performance scores inline in the conversation.
- **Documents:** upload PDF, DOCX, TXT, or Markdown files; view indexed filenames/chunk counts and delete documents.
- **Chunking:** choose basic, recursive-character, sentence, paragraph, semantic, agentic/LLM, or sliding-window chunking, with per-upload strategy-specific boundaries.
- **Hybrid retrieval:** reciprocal-rank fusion over dense FAISS and BM25 keyword results.
- **Persistence:** local FAISS index and JSON chunk metadata in `data/faiss`. Chat history and inline evaluation results stay in the active browser tab session and are not stored by the backend.
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
cd frontend
npm install
npm run dev
```

Open the dev frontend at <http://localhost:5173>. Vite proxies `/api` requests to the local FastAPI server.

## Configuration

Configure `GROK_API_KEY` with a Groq API key, `GROK_BASE_URL` with `https://api.groq.com` (the Groq SDK adds `/openai/v1`), and set `GROK_MODEL` plus `GROK_FALLBACK_MODEL` to model IDs available to your Groq account. The backend retries the configured fallback only for model-unavailable responses. `VITE_API_BASE_URL` optionally configures the frontend API base URL; by default it uses `/api/v1`, with Vite proxying requests to FastAPI during development. `HUGGINGFACE_ENDPOINT` configures the Hugging Face Hub endpoint, and `HUGGINGFACE_MODEL_URL` selects the embedding model; the configured URL is used to derive its Hugging Face repository ID. `HF_TOKEN` is optional for public models and can authenticate access to private or gated repositories. `DATA_DIR` controls local persistence; `CHUNK_SIZE` and `CHUNK_OVERLAP` provide server defaults for API clients that omit per-upload values, and retrieval count is configurable. The default vector store is FAISS. For production deployments, put the API behind an authenticated gateway, restrict network exposure, and use durable persistent storage for the FAISS data.

Character-based chunk sizes, minimums, overlap, and sliding-window strides are measured in characters. Semantic chunking uses the configured embedding model and a configurable cosine-similarity threshold in the range 0–1. Sentence and paragraph chunking can merge short units up to the configured minimum while respecting the maximum where possible. Agentic chunking exposes the configured Groq model, a custom prompt, a maximum token count, and a document-structure preservation option. Token counts use the selected model's `tiktoken` encoding when available and `cl100k_base` otherwise. Agentic chunking sends document text to Groq, may incur API usage, and is limited to 50,000 extracted characters per document. Sliding-window stride must be positive and no larger than the window size; overlap is derived as `window size - stride`.

Use **Evaluate this answer** beneath any assistant response to run RAGAS or DeepEval inline in the conversation. An expected answer is optional; without one, correctness and RAGAS context precision/recall are shown as unavailable. Relevant passages are also optional; retrieval precision/recall, MRR, and NDCG are unavailable without them. RAGAS scores faithfulness and answer relevance, plus correctness and context metrics when an expected answer is provided. DeepEval scores faithfulness, answer relevance, hallucination, and toxicity, plus correctness when an expected answer is provided. Cost is an estimate based on chatbot input/output tokens only; judge-model usage is not included. Set `GROK_INPUT_COST_PER_MILLION` and `GROK_OUTPUT_COST_PER_MILLION` in `.env` to show that estimate.

## API

- `GET /api/v1/health` — liveness check
- `GET /api/v1/ready` — API readiness check
- `GET /api/v1/documents` — list indexed files
- `GET /api/v1/documents/chunking-options` — list the configured embedding and LLM model options
- `POST /api/v1/documents/upload` — ingest one supported file (20 MB max); multipart fields configure the selected strategy, size thresholds, overlap, semantic similarity, agentic prompt/token limit/structure preservation, and sliding window/stride
- `DELETE /api/v1/documents/{document_id}` — delete its FAISS chunks
- `POST /api/v1/chat` — answer using retrieved context; optional history is supplied transiently by the client and is not stored by the API
- `POST /api/v1/evaluation/run` — evaluate test cases with RAGAS or DeepEval

This starter does not include user authentication or multi-tenant isolation; add those before exposing it to untrusted users. Chat history exists only in the current browser tab session and is lost when that session ends.
