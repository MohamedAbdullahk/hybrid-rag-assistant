# 🤖 Conversational Hybrid RAG Assistant

Upload a PDF and chat with it. Answers are grounded in the document, cite their sources, and every step is measured, evaluated and traced.

**Live demo:** https://aimgrow.tech

The system combines **section-wise chunking**, a **FAISS vector store**, a **Knowledge Graph**, **BM25 keyword search**, **Reciprocal Rank Fusion**, a **Cross-Encoder reranker**, conversation **memory**, **guardrails**, an **LLM-as-a-Judge evaluation dashboard** and **LangSmith tracing**. It is containerised with Docker and served behind Nginx with HTTPS.

---

## ✨ Features

| Area | What it does |
|---|---|
| 📄 **PDF processing** | Reads all pages as one text, splits it **at section headings** (`1.`, `7)`, `A.`), prefixes every chunk with a `[Section: ...]` label and keeps page and section metadata |
| 🛑 **Image-PDF guard** | Rejects scanned or image-only PDFs (fewer than 100 readable characters per page) with a clear error, before any embedding cost |
| 🔢 **Vector store** | OpenAI `text-embedding-3-small` embeddings in a FAISS index saved per session |
| 🕸️ **Knowledge Graph** | `gpt-4o-mini` extracts (subject, relation, object) triples from every chunk (**8 parallel calls**), stored as a NetworkX graph |
| 🔍 **Hybrid retrieval** | FAISS (meaning) + BM25 (exact keywords), merged with **RRF**, reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`, plus matching graph relationships |
| 🧠 **Memory** | Keeps the last 6 turns per session and rewrites follow-up questions into standalone questions |
| 🛡️ **Guardrails** | Blocks prompt-injection patterns on input and redacts secrets (API keys) on output |
| 💬 **Small-talk router** | Answers greetings instantly without calling the LLM |
| 📊 **Evaluation dashboard** | Runs every RAG strategy on a test question and scores **Faithfulness, Answer Relevancy and Context Relevancy** with an LLM judge; shows a table, a chart, PASS/FAIL and the best strategy |
| 🔭 **Observability** | LangSmith traces every upload, chat and evaluation as one tree (FAISS, BM25, RRF, rerank, KG, LLM), with latency and token cost. Stage timings are also logged |
| 🚀 **Deployment** | Docker Compose, Nginx reverse proxy, Let's Encrypt SSL, Hostinger VPS |

---

## 🏗️ Architecture

```
                         ┌──────────────── INGEST ────────────────┐
  PDF ──► Image-PDF guard ──► Section-wise chunking ──┬─► Embeddings ──► FAISS index
                                                      └─► LLM triple extraction (8 parallel) ──► Knowledge Graph

                         ┌──────────────── ANSWER ────────────────┐
  Question ──► Guardrails (input) ──► Small-talk router ──► Query rewrite (memory)
           ──► Hybrid retrieval: FAISS + BM25 ──► RRF ──► Cross-Encoder rerank ──► top 4 chunks
                                  + Knowledge Graph relationships
           ──► gpt-4o-mini answer (with sources) ──► Guardrails (output) ──► Streamlit UI

                         ┌──────────── EVALUATE & OBSERVE ────────────┐
  Evaluation: every strategy ──► LLM judge (faithfulness / relevancy / context) ──► best strategy
  LangSmith: one trace tree per upload, chat and evaluation · app.log: stage timings
```

### RAG strategies

| Strategy | Pipeline |
|---|---|
| `baseline` | FAISS vector search (top 4) + Knowledge Graph |
| `hybrid_rerank` *(default)* | FAISS + BM25 → RRF → Cross-Encoder rerank (top 4) + Knowledge Graph |
| `fusion` | *Placeholder:* currently runs the `hybrid_rerank` pipeline (multi-query generation is on the roadmap) |
| `crag` | *Placeholder:* currently runs the `hybrid_rerank` pipeline (corrective RAG is on the roadmap) |

Set the default with `RAG_STRATEGY` in `.env`.

---

## 📈 Measured improvements

Real numbers from the evaluation dashboard and logs on the sample `spotify_web_app_architecture.pdf` (6 pages):

| Change | Before | After |
|---|---|---|
| Section-wise chunking, question *"Which service owns the playback_queue table?"* | 0.67 ❌ FAIL (answer mixed in a second service) | **1.00 ✅ PASS** |
| Knowledge Graph prompt fix (escaped `{{ }}` + domain-neutral prompt) | 0 nodes | **321 nodes, 318 edges** |
| Parallel KG extraction (8 calls, all chunks) | 52 s upload, 15 chunks in KG | **~17–23 s upload, all 37 chunks in KG** |

> One question is not a benchmark. Use 10–20 questions with expected answers before choosing a strategy.

---

## 🧪 Evaluation: how to use it

The dashboard is an **offline benchmark**: give a question and (ideally) the expected answer, and every strategy is run and judged.

| Metric | Question the judge answers |
|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved chunks? (catches hallucination) |
| **Answer Relevancy** | Does the answer address the question and agree with the expected answer? |
| **Context Relevancy** | Did retrieval bring the chunks needed to answer? |

- **Overall** = the average of the three. **PASS** if Overall ≥ **0.7**.
- **Best strategy** = the highest Overall score. On a tie, the fastest strategy wins, and the UI says so.
- Always give an **expected answer**: without one, the judge is more lenient.
- Workflow: *measure → read the judge's reason → fix → re-measure.*

---

## 📁 Project structure

```
Hybrid_rag/
├── backend/
│   ├── main.py                  # FastAPI app and routers
│   ├── config.py                # Settings from .env (+ LangSmith env setup)
│   ├── rag/
│   │   ├── pdf_processor.py     # Section-wise chunking + image-PDF guard
│   │   ├── vector_store.py      # FAISS index per session (OpenAI embeddings)
│   │   ├── bm25_retriever.py    # BM25 keyword search
│   │   ├── reranker.py          # Cross-Encoder reranker
│   │   └── strategies.py        # baseline / hybrid_rerank / fusion / crag + RRF (traced)
│   ├── kg/
│   │   ├── extractor.py         # LLM triple extraction
│   │   └── graph_store.py       # NetworkX graph build, save, search
│   ├── services/
│   │   ├── rag_service.py       # Guardrails → router → rewrite → retrieve → answer (traced)
│   │   ├── guardrails.py        # Input injection checks, output redaction
│   │   └── intent_router.py     # Small-talk fast replies
│   ├── memory/session_memory.py # Last-6-turn memory per session
│   ├── eval/evaluator.py        # LLM-as-a-Judge, strategy comparison
│   ├── routes/
│   │   ├── document.py          # Upload + background indexing (parallel KG)
│   │   ├── chat.py              # Chat endpoint
│   │   ├── evaluation.py        # Evaluation endpoint
│   │   └── diagnostics.py       # Latency statistics from logs
│   └── utils/                   # logger, stage timer
├── frontend/app.py              # Streamlit chat + evaluation dashboard
├── docker/Dockerfile.backend
├── docker/Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

Per-session data is saved under `data/<session_id>/`: the uploaded PDF, `faiss_index/` and `knowledge_graph.graphml`.

---

## 🔌 API

Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/v1/document/upload` | Upload a PDF (multipart `file`); returns `session_id`, indexing runs in the background |
| `GET` | `/api/v1/document/status/{session_id}` | Indexing status: `processing` / `completed` / `failed` |
| `POST` | `/api/v1/chat/` | `{"session_id": "...", "message": "..."}` → answer, sources, timings |
| `POST` | `/api/v1/eval/compare` | `{"session_id": "...", "question": "...", "expected_answer": "..."}` → scores for every strategy + best strategy |
| `GET` | `/api/v1/diagnostics/latency` | Average / min / max latency per stage |

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` to `.env` and fill in the values. **Never commit `.env`**; it is in `.gitignore`.

| Variable | Example | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | Required |
| `LLM_MODEL` | `gpt-4o-mini` | Used for answers, KG extraction and the judge |
| `RAG_STRATEGY` | `hybrid_rerank` | Default strategy for chat |
| `DATA_DIR` / `LOG_DIR` | `./data` / `./logs` | |
| `LANGCHAIN_TRACING_V2` | `true` | LangSmith tracing (optional) |
| `LANGCHAIN_ENDPOINT` | `https://api.smith.langchain.com` | Use `https://eu.api.smith.langchain.com` for EU accounts |
| `LANGCHAIN_API_KEY` | `lsv2_...` | Tracing is enabled only when this is set |
| `LANGCHAIN_PROJECT` | `hybrid-rag-production` | Project name in LangSmith |

---

## 💻 Run locally (Windows CMD)

```cmd
cd Hybrid_rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Then open `.env` and fill in your keys.

**Terminal 1: backend**
```cmd
venv\Scripts\activate
python -m backend.main
```
Use `python -m backend.main` from the project root, not `python backend/main.py`.

**Terminal 2: frontend**
```cmd
venv\Scripts\activate
set "BACKEND_HOST=http://localhost:8000"
streamlit run frontend\app.py
```
Open http://localhost:8501. Without `BACKEND_HOST`, the frontend looks for the Docker hostname `hybrid_rag_backend` and fails locally.

On macOS or Linux, use `source venv/bin/activate` and `export BACKEND_HOST=http://localhost:8000`.

---

## 🐳 Run with Docker

```bash
docker compose up -d --build
docker compose ps
```

- The backend (`8000`) and frontend (`8501`) are published on **`127.0.0.1` only**, so they are reachable from the server itself (Nginx) and not directly from the internet. Docker bypasses `ufw`, so this binding is what keeps the ports private.
- Inside the containers the apps still listen on `0.0.0.0`, as Docker requires.
- `./data` is mounted into the backend, so sessions survive rebuilds. `./logs` is mounted too.
- The frontend reaches the backend over the Docker network at `http://hybrid_rag_backend:8000`.

---

## 🌐 Production deployment (VPS + Nginx + SSL)

1. Push to GitHub, then on the VPS: `git pull && docker compose up -d --build`
2. Nginx site (`/etc/nginx/sites-available/<domain>`), simplified:

```nginx
server {
    server_name yourdomain.com;

    location / {                                   # Streamlit (needs WebSocket)
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    location /docs         { proxy_pass http://localhost:8000/docs; }          # optional
    location /openapi.json { proxy_pass http://localhost:8000/openapi.json; }  # optional
}
```

3. `sudo nginx -t && sudo systemctl reload nginx`
4. HTTPS: `sudo certbot --nginx -d yourdomain.com`
5. Firewall: `sudo ufw allow OpenSSH && sudo ufw allow 'Nginx Full' && sudo ufw enable`

**Check that the ports are private:** from your own laptop, `curl http://<VPS-IP>:8000/` should **fail**. On the VPS, `curl http://localhost:8000/` should return `{"status":"online", ...}`.

---

## 🔭 Observability with LangSmith

With `LANGCHAIN_API_KEY` set, open smith.langchain.com → **Tracing** → your project. Each operation is one tree:

```
answer_question
 ├─ rewrite_query
 ├─ retrieve_context
 │   ├─ faiss_search · kg_search · bm25_search · rrf_fusion · cross_encoder_rerank
 └─ ChatOpenAI            (prompt, answer, tokens, cost)

process_pdf
 ├─ pdf_chunking
 └─ kg_extract_triples → ChatOpenAI   (one per chunk, run in parallel)

compare_all_strategies
 └─ evaluate_strategy → answer_question … → llm_judge
```

Use it to see which step is slow and what the LLM actually received and returned. Worker threads get a copy of the trace context, so parallel KG calls stay under `process_pdf`.

---

## ⚠️ Known limitations

- **Scanned / image PDFs** are rejected; OCR (Tesseract) is not included yet.
- `fusion` and `crag` are placeholders that currently run the `hybrid_rerank` pipeline.
- **Sessions, processing status and chat memory are held in memory.** A backend restart clears them (indexes on disk remain). While a PDF is being indexed, a restart leaves the UI polling a status that no longer exists; refresh and upload again.
- Knowledge Graph search is simple keyword matching over edges (top 10).
- Evaluation is one question at a time; batch evaluation over a question set is on the roadmap.
- `deepeval` is still listed in `requirements.txt` but is **no longer used**; evaluation uses the built-in LLM judge.

## 🗺️ Roadmap

- [ ] Batch evaluation over a saved question set, with average scores
- [ ] Real `fusion` (multi-query) and `crag` (corrective RAG) strategies
- [ ] Live per-answer evaluation badge in chat (reference-free metrics)
- [ ] Hide sources when the answer is "I couldn't find this in the document"
- [ ] OCR fallback for image PDFs
- [ ] Persist sessions and status (e.g. SQLite / Redis)
- [ ] Remove unused dependencies and pin the frontend Streamlit version

---

## 🧰 Tech stack

**Backend:** FastAPI · LangChain · OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) · FAISS · rank-bm25 · sentence-transformers Cross-Encoder · NetworkX · pypdf · LangSmith
**Frontend:** Streamlit · pandas · Altair
**Ops:** Docker Compose · Nginx · Let's Encrypt (Certbot) · Hostinger VPS · GitHub

---

Built by **Abdullah**.
