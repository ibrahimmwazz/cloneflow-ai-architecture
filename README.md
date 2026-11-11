# CloneFlow AI — Serverless GenAI, Hybrid RAG & Agentic Video Architecture

> Production AI engine and orchestration showcase for **CloneFlow AI** (Live App: [ai-clone-tan-six.vercel.app](https://ai-clone-tan-six.vercel.app) | Live API Health: [ai-clone.fly.dev/health](https://ai-clone.fly.dev/health))

This repository showcases the core AI/ML, Hybrid RAG, multi-step AI agent, Responsible AI guardrails, GPU optimization, and backend orchestration architecture powering CloneFlow AI. *(Frontend UI and commercial workspace assets remain in the private SaaS repository.)*

---

## Architecture Overview

| Layer | Technologies & Implementation |
|---|---|
| **Backend Orchestrator** | FastAPI on Fly.io ([`Dockerfile.api`](Dockerfile.api), [`fly.toml`](fly.toml)), Pydantic v2, SlowAPI rate limiting, HMAC worker auth, Sentry |
| **Hybrid RAG & Vector DB** | PostgreSQL `pgvector` HNSW index + RLS + `match_brand_chunks` RPC ([`database/schema.sql`](database/schema.sql)), 384-d embeddings (`BAAI/bge-small-en-v1.5` / OpenAI / deterministic hash fallback), Reciprocal Rank Fusion (RRF) ([`api/services/rag_service.py`](api/services/rag_service.py)) |
| **Tool-Using AI Agent** | 4-tool [`SceneDirectorAgent`](api/services/director_agent.py) (`guardrail_scan` → `retrieve_brand_knowledge` → `compose_storyboard` → `verify_and_critique`) with citations and bounded self-critique refinement |
| **Responsible AI Guardrails** | [`AIGuardrails`](api/services/ai_guardrails.py) blocking prompt injections, role hijacking, XSS/SQL smuggling, and fraud prompts while redacting PII and credentials; C2PA provenance + atomic PostgreSQL credit RPCs |
| **LLMOps Evaluation** | Automated SLA benchmark engine ([`api/services/ai_evaluator.py`](api/services/ai_evaluator.py), [`scripts/eval_ai_pipeline.py`](scripts/eval_ai_pipeline.py)) with optional `agent_eval_runs` persistence |
| **Serverless GPU Workers** | Modal Labs Nvidia T4 / L4 / A10G pipelines ([`modal_worker.py`](modal_worker.py)) orchestrating MuseTalk, LivePortrait, SadTalker, Fish Speech 1.5, Whisper large-v3, NLLB-200, Wav2Lip, FLUX.1-schnell, CogVideoX-5B, and HunyuanVideo |
| **GPU Inference Optimizations** | `bitsandbytes` 8-bit quantization, `bfloat16` precision, `torch.compile(mode="max-autotune")`, PyTorch SDPA attention, VAE slicing & tiling, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, sequential CPU offloading |

---

## Repository Structure

```text
├── api/
│   ├── main.py                      # FastAPI entrypoint, middleware stack, CORS & routers
│   ├── config.py                    # Fail-closed Pydantic settings
│   ├── dependencies/                # Supabase JWT & auth dependencies
│   ├── middleware/                  # Validation, sanitization & credit rate limiting
│   ├── routes/
│   │   ├── rag_agent.py             # /api/rag/* and /api/agent/* REST endpoints
│   │   ├── jobs.py                  # Async GPU job lifecycle & HMAC webhooks
│   │   ├── video_gen.py             # CogVideoX / HunyuanVideo generation endpoints
│   │   ├── realtime.py              # Resilient SSE status streaming
│   │   └── ...                      # Auth, avatars, dubbing, image_gen, billing
│   ├── services/
│   │   ├── rag_service.py           # Sliding-window chunker, 384-d embeddings, Hybrid RRF
│   │   ├── director_agent.py        # Multi-step tool-using Scene Director Agent
│   │   ├── ai_guardrails.py         # Prompt-injection firewall & secret/PII redactor
│   │   ├── ai_evaluator.py          # Automated RAG/Agent/Guardrail benchmark runner
│   │   └── ...                      # Supabase client, Redis queues, media services
│   └── tests/
│       └── test_rag_agent_eval.py   # End-to-end unit & FastAPI integration suite
├── database/
│   └── schema.sql                   # PostgreSQL + pgvector HNSW schema, RLS & atomic RPCs
├── scripts/
│   └── eval_ai_pipeline.py          # CI-ready LLMOps benchmark CLI
├── .github/workflows/ci.yml         # CI/CD (PyTest, LLMOps Eval CLI & Trivy Security Scan)
├── modal_worker.py                  # Serverless GPU inference workers (Modal Labs)
├── Dockerfile.api                   # Production API container
└── fly.toml                         # Fly.io deployment manifest
```

---

## Quick Verification (Runs Offline Without External API Keys)

```bash
# 1. Install backend dependencies
pip install -r api/requirements.txt pytest

# 2. Run RAG, Agent, Guardrails, Evaluator & FastAPI middleware integration tests
pytest

# 3. Run the automated LLMOps Evaluation CLI
python scripts/eval_ai_pipeline.py
```

Latest verified benchmark output:
- `retrieval_hit_at_3`: `1.0` | `retrieval_keyword_recall`: `1.0`
- `schema_compliance_rate`: `1.0` | `model_routing_accuracy`: `1.0`
- `mean_grounding_score`: `0.9667` | `mean_pacing_error`: `0.24`
- `injection_block_rate`: `1.0` | `pii_redaction_rate`: `1.0`

---

## License

MIT License
