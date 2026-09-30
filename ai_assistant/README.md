# Acme AI Assistant (W15 Assignment)

A RAG-based AI assistant that answers questions about a (fictional) company's HR policy,
IT support guide and product FAQ. It uses **Groq** for the LLM, **MiniLM** embeddings from
Hugging Face, **ChromaDB** as the vector database, supports **tool calling** and always returns
**structured JSON**. Task 2 adds a **Streamlit UI**, **ONNX optimization**, **async/batch processing**,
**caching**, **retries**, **rate limiting**, **fallback models** and **Docker Compose** deployment.

- Task 1 architecture: [../arch_diagram_1.png](../arch_diagram_1.png)
- Task 2 architecture: [../arch_diagram_2.png](../arch_diagram_2.png)
- Quick start: [../how_to_run.txt](../how_to_run.txt)

---

## 1. Tech stack

| Part | What I used | Why |
|------|-------------|-----|
| LLM provider | Groq (`openai/gpt-oss-120b`) | Free tier, very fast, OpenAI-compatible API with tool calling |
| Fallback LLM | Groq `llama-3.1-8b-instant`, then local vLLM | Smaller model has separate rate limits; vLLM works without internet |
| Local model | vLLM serving `Qwen2.5-1.5B-Instruct` | Open-source, small enough for one GPU, supports tool calling (hermes parser) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small (90 MB), fast on CPU, good quality for short documents |
| Vector DB | ChromaDB (persistent, cosine) | Runs locally, no extra server needed |
| Backend | FastAPI (async) | Async support for concurrent requests, automatic docs at `/docs` |
| UI | Streamlit | Quick to build a chat interface |
| Optimization | ONNX Runtime + INT8 dynamic quantization | 7x faster embedding on CPU |

---

## 2. Project structure

```
W15_Submission/
├── how_to_run.txt         # short run instructions
├── arch_diagram_1.png     # Task 1 architecture
├── arch_diagram_2.png     # Task 2 architecture
├── docker-compose.yml     # api + ui (+ vllm) services, builds ./ai_assistant
└── ai_assistant/
    ├── app/
    │   ├── config.py          # all settings (read from .env)
    │   ├── prompts.py         # system prompt, JSON format, prompt engineering notes
    │   ├── schemas.py         # pydantic models (LLM JSON output + API request/response)
    │   ├── llm_client.py      # Groq/vLLM client, retry with backoff, fallback providers, semaphore
    │   ├── tools.py           # tools for function calling + their JSON schemas
    │   ├── assistant.py       # RAG + tool calling loop + JSON validation + graceful degradation
    │   ├── cache.py           # TTL + LRU response cache
    │   ├── rate_limiter.py    # sliding window rate limiter per IP
    │   ├── api.py             # FastAPI app (/chat, /chat/batch, /health, /ingest)
    │   ├── cli.py             # terminal chat (Task 1 demo)
    │   └── rag/
    │       ├── ingest.py      # load files -> chunk -> embed -> store
    │       ├── embeddings.py  # MiniLM with PyTorch or ONNX Runtime backend
    │       └── vector_store.py# ChromaDB add / search
    ├── ui/streamlit_app.py    # web UI (chat tab + batch tab)
    ├── scripts/
    │   ├── export_onnx.py         # convert MiniLM to ONNX + INT8 and verify outputs
    │   ├── benchmark_embeddings.py# PyTorch vs ONNX speed test
    │   ├── eval_retrieval.py      # retrieval hit-rate test used to pick chunk size
    │   ├── load_test.py           # concurrent load test for the API
    │   └── start_vllm.sh          # serve a local open-source LLM with vLLM
    ├── data/                  # documents for RAG (.md, .txt, .pdf)
    ├── tests/test_basic.py    # unit tests (no API key needed)
    ├── docs/                  # mermaid source of the architecture diagrams
    ├── Dockerfile
    ├── requirements.txt
    └── .env.example
```

---

## 3. Setup

### 3.1 Get a Groq API key
Create a free key at https://console.groq.com/keys, then:

```bash
cp ai_assistant/.env.example ai_assistant/.env
# open ai_assistant/.env and set GROQ_API_KEY=gsk_...
```

### 3.2 Run with Docker Compose (recommended)

Run from the `W15_Submission/` folder (where `docker-compose.yml` is):

```bash
docker compose up --build
```

- UI: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs
- Health: http://localhost:8000/health

The first build takes a while (it downloads CPU torch and the MiniLM model and exports the ONNX model).
Documents in `data/` are ingested automatically on the first start.

With a local vLLM model as the last fallback (needs an NVIDIA GPU + nvidia-container-toolkit):

```bash
# in ai_assistant/.env: VLLM_BASE_URL=http://vllm:8000/v1
docker compose --profile gpu up --build
```

Useful commands:

```bash
docker compose logs -f api                                   # see retries / fallbacks in the logs
docker compose exec api pytest -q                            # run tests inside the container
curl -X POST "http://localhost:8000/ingest?reset=true"       # re-ingest after changing data/
docker compose down                                          # stop (add -v to delete the vector DB)
```

### 3.3 Run locally without Docker (Python 3.11 or 3.12)

```bash
cd ai_assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m app.rag.ingest --reset   # build the vector DB
python -m app.cli                  # Task 1: terminal chat

# Task 2: API + UI (two terminals)
uvicorn app.api:app --port 8000
streamlit run ui/streamlit_app.py

# optional: ONNX embeddings
python scripts/export_onnx.py
# set USE_ONNX_EMBEDDINGS=true in .env, then re-ingest:
python -m app.rag.ingest --reset
```

> If you switch between the PyTorch and ONNX embedding backends, run the ingestion again with
> `--reset`. The INT8 vectors are slightly different from the PyTorch ones.

---

## 4. Task 1 – AI Assistant

### 4.1 LLM integration
`app/llm_client.py` uses the `openai` Python SDK with `base_url=https://api.groq.com/openai/v1`.
Because Groq and vLLM both speak the OpenAI API, the same code works for the hosted model and the local model.

### 4.2 Prompt engineering and parameters
The system prompt is in `app/prompts.py`. It:
- gives the assistant a role and scope (Acme HR / IT / product questions),
- tells it to answer **only from the context** and to say "I don't know" instead of making things up,
- explains when to use each tool,
- gives the exact JSON output format.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `temperature` | 0.2 | factual Q&A, we want the same answer every time |
| `top_p` | 0.9 | cuts very unlikely tokens but still allows natural wording |
| `max_tokens` | 2000 | gpt-oss-120b is a reasoning model and its reasoning tokens count in this limit. With 800 the final answer could be cut off |
| `reasoning_effort` | low | (gpt-oss only) document Q&A doesn't need deep reasoning; low is faster and uses fewer tokens |

Both can be changed per request (API field or UI sliders) to compare the behaviour.

### 4.3 Structured output (JSON)
Every answer follows this schema (`AssistantAnswer` in `app/schemas.py`):

```json
{
  "answer": "You get 24 days of paid annual leave per year, accrued at 2 days per month.",
  "sources": ["acme_hr_policy.md"],
  "confidence": "high",
  "follow_up_questions": ["How many leave days can I carry forward?"]
}
```

To make sure it is valid:
1. The prompt asks for JSON only.
2. `parse_json_answer()` removes markdown fences and validates with **pydantic**.
3. If it is still invalid, a second "repair" call is made with `response_format={"type": "json_object"}` (Groq JSON mode).
4. If that also fails, the raw text is wrapped into the schema with `confidence: "low"`.

### 4.4 Tool calling
Tools are defined in `app/tools.py` in the OpenAI function-calling format:

| Tool | What it does |
|------|--------------|
| `search_documents` | lets the model run another vector search with its own query |
| `calculator` | safe maths using Python `ast` (no `eval`) |
| `get_current_datetime` | current date/time in any timezone |
| `get_weather` | real external API call (wttr.in) |
| `create_support_ticket` | "action" tool: saves an IT ticket to `tickets.json` |

The loop in `assistant.py` sends the tools, runs any `tool_calls` the model returns, sends the results back
and repeats (max 3 rounds) until the model gives the final answer.

Example questions that trigger tools:
- "I have used 7 of my annual leaves, how many are left?" → `calculator`
- "My laptop won't boot, please raise a ticket" → `create_support_ticket`
- "What's the weather in Mumbai today?" → `get_weather`

### 4.5 RAG pipeline
1. **Load** `.md`, `.txt` and `.pdf` files from `data/` (`pypdf` for PDFs).
2. **Chunk**: split on paragraphs, merge up to 500 characters with 100 characters overlap. Markdown headings
   stay attached to the paragraph below them (at first, headings like `## 3. Leave Policy` ended up at the end
   of the previous chunk and retrieval for leave questions was bad).
3. **Embed** with `all-MiniLM-L6-v2` (normalized 384-d vectors, batch size 32).
4. **Store** in ChromaDB with cosine distance. `upsert` with ids like `file.md-3` so re-running ingestion does not create duplicates.
5. **Retrieve** the top 4 chunks for each question and put them into the prompt with their source file name.

**Choosing the chunk size.** `scripts/eval_retrieval.py` asks 15 questions and checks if a chunk containing the
expected answer is in the top-k results:

| chunk size / overlap | chunks | hit@2 | hit@4 |
|----------------------|--------|-------|-------|
| 300 / 50  | 30 | 14/15 | 14/15 |
| 400 / 80  | 25 | 13/15 | 14/15 |
| **500 / 100** | **20** | **14/15** | **15/15** |
| 800 / 150 | 11 | 14/15 | 15/15 |

500/100 with top-4 found every answer while keeping the prompt smaller than 800-character chunks.

### 4.6 Local deployment with vLLM
`scripts/start_vllm.sh` (or the `vllm` service in docker compose) serves `Qwen/Qwen2.5-1.5B-Instruct`:

```bash
docker run --gpus all -p 8001:8000 --ipc=host vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Then set `VLLM_BASE_URL=http://localhost:8001/v1` in `.env`. The assistant uses it as the last fallback.
To use **only** the local model, leave `GROQ_API_KEY` empty.

> vLLM needs an NVIDIA GPU (it does not run on Apple Silicon). A free Colab/Kaggle T4 or a cloud GPU VM works for testing.

### 4.7 Containerization
`Dockerfile`: `python:3.12-slim`, CPU-only torch (smaller image), the MiniLM model is downloaded at build time
(the container works offline and starts faster), the ONNX export runs at build time and the app runs as a non-root user.

```bash
cd ai_assistant
docker build -t acme-ai-assistant .
docker run -p 8000:8000 --env-file .env acme-ai-assistant
```

---

## 5. Task 2 – Productionizing

### 5.1 Web UI
`ui/streamlit_app.py` talks to the backend only through HTTP (`API_URL`).
- **Chat tab**: chat history, sources, tools used, model used, latency, cache hit, raw JSON.
- **Batch tab**: send up to 10 questions at once.
- **Sidebar**: temperature / top_p sliders, cache on/off, health check, re-ingest button.
- Clear messages for errors: backend down, timeout, rate limited (429), degraded mode.

### 5.2 Model optimization (ONNX)
**What was converted:** the MiniLM embedding model (`scripts/export_onnx.py`), because it runs inside
our container on CPU for every request and every ingested chunk.

**What was not converted and why:** the LLM.
- The main LLM runs on Groq's servers, so we don't have its weights.
- The local LLM runs on vLLM, which already uses optimized GPU kernels (PagedAttention, continuous batching,
  CUDA graphs). An ONNX export of a billion-parameter decoder would not be faster than vLLM on GPU.
- Llama/Qwen models are also not trained by me in this project, so there is nothing custom to export.

**Optimizations applied:**
1. Export to ONNX (opset 17, dynamic batch and sequence length) and run with ONNX Runtime.
2. INT8 **dynamic quantization** (`onnxruntime.quantization.quantize_dynamic`).
3. The output is checked against the original PyTorch model.

Results on my laptop (Apple M2 CPU, `python scripts/benchmark_embeddings.py`):

| Backend | Model size | Single query p50 | Batch of 32 | Cosine sim. vs PyTorch | Retrieval hit@4 |
|---------|-----------:|-----------------:|------------:|-----------------------:|----------------:|
| PyTorch (sentence-transformers) | 90 MB | 5.78 ms | 23.6 ms | 1.00 | 15/15 |
| ONNX FP32 | 90 MB | 1.63 ms | 34.6 ms | 1.00 | 15/15 |
| **ONNX INT8 (used in Docker)** | **23 MB** | **0.81 ms** | **15.6 ms** | 0.96–0.98 | **15/15** |

INT8 is ~7x faster for a single query and 4x smaller, with the same retrieval result on our test questions.
(Numbers change between machines. Run the script to get your own.)

### 5.3 Performance engineering

| Technique | Where | Effect |
|-----------|-------|--------|
| Async endpoints + `AsyncOpenAI` | `api.py`, `llm_client.py` | while one request waits for Groq, the server handles other requests |
| CPU work in threads (`asyncio.to_thread`) | `assistant.py` | embedding/search and tools don't block the event loop |
| Batch endpoint `/chat/batch` | `api.py` | questions run concurrently with `asyncio.gather` |
| Semaphore (max 5 LLM calls) | `llm_client.py` | protects against hitting Groq rate limits when many users come at once |
| Response cache (TTL 10 min, LRU 200) | `cache.py` | repeated questions return in ~1 ms and use no API quota |
| ONNX INT8 embeddings | `embeddings.py` | faster retrieval |
| Short history (last 6 messages), top-4 chunks, `reasoning_effort=low` | `assistant.py`, `config.py` | smaller prompts = lower latency |
| Fast provider (Groq LPU) | `config.py` | low time-to-first-token |

**Load test:**

```bash
python scripts/load_test.py --requests 20 --concurrency 5 --no-cache   # real LLM calls
python scripts/load_test.py --requests 20 --concurrency 5              # with cache
```

It prints success / cached / degraded / 429 counts, p50/p95 latency and requests per second.
Run it with your Groq key and add the numbers to this section.
Note: the default rate limit is 20 requests/minute per IP, so set `RATE_LIMIT_PER_MINUTE=1000` in `.env` for bigger load tests.

### 5.4 Reliability

| Requirement | Implementation |
|-------------|----------------|
| **Retry** | `tenacity`: 3 attempts, exponential backoff (1s, 2s, 4s… max 8s), only for temporary errors (429 from provider, timeout, connection error, 5xx). A wrong API key (401) or a bad request (400) is not retried. |
| **Rate limiting** | Sliding window per client IP (default 20/min). Returns HTTP 429 with a `Retry-After` header. |
| **Fallback model/provider** | `groq/openai/gpt-oss-120b` → `groq/llama-3.1-8b-instant` → local vLLM. The response has a `model_used` field so you can see which one answered. |
| **Error handling** | Pydantic validation of requests (422), 30s LLM timeout, tool errors are sent back to the LLM as text, invalid JSON is repaired, global exception handler returns a clean 500 message (no stack trace). |
| **Graceful degradation** | If all LLMs fail, the API still returns 200 with `degraded: true` and the most relevant document passages. If the vector DB fails, the LLM answers without context. Degraded answers are not cached. `/health` reports `degraded` status. |

What I tested without a Groq key:
- API without a key → `/health` shows `degraded`, `/chat` returns the document passages with `degraded: true`.
- Invalid key + unreachable vLLM URL → the logs show 401 on both Groq models (moved on without retry), then 3 attempts with backoff for vLLM, then the degraded answer (~9 s).
- 4 fast requests with `RATE_LIMIT_PER_MINUTE=5` → `200 200 200 429`.
- Streamlit UI with the backend down → shows "Cannot connect to the backend" instead of crashing.

### 5.5 Tests

```bash
pytest -q
```

12 unit tests: chunking, calculator safety, JSON parsing, cache expiry/LRU, rate limiter and graceful degradation (LLM mocked).

---

## 6. API reference

```bash
# ask a question
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"question": "How many paid leaves do I get per year?", "temperature": 0.2, "top_p": 0.9}'

# with chat history
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"question": "And sick leave?", "history": [{"role": "user", "content": "How many paid leaves do I get?"}, {"role": "assistant", "content": "24 days per year."}]}'

# batch
curl -X POST http://localhost:8000/chat/batch -H "Content-Type: application/json" \
  -d '{"questions": ["Price of the Business plan?", "How do I reset my password?"]}'
```

Response:

```json
{
  "answer": "...",
  "sources": ["acme_hr_policy.md"],
  "confidence": "high",
  "follow_up_questions": ["..."],
  "model_used": "groq/openai/gpt-oss-120b",
  "tools_used": [],
  "cached": false,
  "degraded": false,
  "latency_ms": 812
}
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | providers, vector DB status, chunk count, embedding backend, cache stats |
| POST | `/chat` | one question (rate limited, cached) |
| POST | `/chat/batch` | up to 10 questions processed concurrently |
| POST | `/ingest?reset=true` | re-ingest `data/` and clear the cache |
| POST | `/cache/clear` | clear the response cache |

---

## 7. Deployment to the cloud (bonus)

### AWS EC2 (simplest)
1. Launch an Ubuntu 24.04 instance (`t3.large`, 2 vCPU / 8 GB RAM, 30 GB disk).
2. Security group: allow port 22 from your IP and port 8501 (UI). Only open 8000 if you need the API from outside.
3. Install Docker and copy the project:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-v2
   sudo usermod -aG docker $USER && newgrp docker
   # copy the W15_Submission folder to the server (git clone or scp), then:
   cd W15_Submission
   cp ai_assistant/.env.example ai_assistant/.env && nano ai_assistant/.env   # add GROQ_API_KEY
   docker compose up -d --build
   ```
4. Open `http://<EC2-public-IP>:8501`.
5. For a local vLLM model, use a GPU instance (`g5.xlarge`, Deep Learning AMI with the NVIDIA drivers) and run `docker compose --profile gpu up -d`.

### Azure Container Apps
```bash
az group create -n acme-ai -l centralindia
az acr create -g acme-ai -n acmeaiacr --sku Basic --admin-enabled true
az acr build -r acmeaiacr -t acme-ai-assistant:latest ./ai_assistant
az containerapp env create -g acme-ai -n acme-env -l centralindia

az containerapp create -g acme-ai -n acme-api --environment acme-env \
  --image acmeaiacr.azurecr.io/acme-ai-assistant:latest --registry-server acmeaiacr.azurecr.io \
  --target-port 8000 --ingress internal --cpu 1 --memory 2Gi \
  --secrets groq-key=<YOUR_KEY> --env-vars GROQ_API_KEY=secretref:groq-key

az containerapp create -g acme-ai -n acme-ui --environment acme-env \
  --image acmeaiacr.azurecr.io/acme-ai-assistant:latest --registry-server acmeaiacr.azurecr.io \
  --target-port 8501 --ingress external \
  --command streamlit --args "run" "ui/streamlit_app.py" "--server.port=8501" "--server.address=0.0.0.0" \
  --env-vars API_URL=http://acme-api
```

(GCP Cloud Run works the same way: push the image to Artifact Registry and deploy two services.)

**Production notes:** store the API key in a secret manager (not in the image), put the UI behind HTTPS
(load balancer or Caddy/Nginx), and add authentication if the assistant is public.

---

## 8. Limitations and future improvements

- Cache and rate limiter are in memory. With more than one API container they are not shared → use **Redis**.
- Answers are not streamed. Streaming tokens to the UI would make them feel faster.
- Retrieval is vector-only. Hybrid search (BM25 + vectors) or a re-ranker would help with exact terms like plan names.
- No user authentication.
- Evaluation is small (15 questions). A bigger test set with an LLM-as-judge would be better.
