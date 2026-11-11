"""
CloneFlow AI — Hybrid RAG Knowledge Base & Scene Director Agent Routes
=======================================================================
Provides authenticated REST endpoints for:
  - Tenant-isolated brand document ingestion into pgvector (`POST /api/rag/ingest`)
  - Hybrid Dense + Lexical RRF retrieval (`POST /api/rag/query`)
  - Document listing & deletion (`GET /api/rag/documents`, `DELETE /api/rag/documents/{doc_id}`)
  - Multi-step tool-using Scene Director Agent (`POST /api/agent/direct`)
  - Automated LLMOps benchmark execution (`POST /api/agent/evaluate`)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies.auth import require_user
from api.services.ai_evaluator import get_ai_evaluator
from api.services.director_agent import get_director_agent
from api.services.rag_service import get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["RAG & AI Agents"])


class RagIngestRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    content: str = Field(..., min_length=15, max_length=24000)
    source_type: str = Field("brand_guide", max_length=50)
    metadata: Optional[Dict[str, Any]] = None


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1500)
    top_k: int = Field(4, ge=1, le=12)
    min_similarity: float = Field(0.05, ge=0.0, le=1.0)


class DirectorAgentRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=2000)
    tone: Literal["professional", "casual", "energetic", "educational"] = "professional"
    target_duration_seconds: int = Field(30, ge=10, le=120)
    aspect_ratio: str = Field("16:9", max_length=16)
    use_rag: bool = True
    top_k: int = Field(4, ge=1, le=10)


class EvaluateRequest(BaseModel):
    persist_results: bool = True
    use_remote_rag: bool = False


@router.post("/rag/ingest")
async def ingest_brand_document(
    body: RagIngestRequest,
    user_id: str = Depends(require_user),
):
    """Sanitize, chunk, embed, and index a brand/knowledge document for Hybrid RAG."""
    rag = get_rag_service()
    try:
        return await rag.ingest_document(
            user_id=user_id,
            title=body.title,
            content=body.content,
            source_type=body.source_type,
            metadata=body.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rag/query")
async def query_brand_knowledge(
    body: RagQueryRequest,
    user_id: str = Depends(require_user),
):
    """Run Hybrid Search (pgvector cosine + lexical overlap via RRF) on user documents."""
    rag = get_rag_service()
    result = await rag.query(
        user_id=user_id,
        query_text=body.query,
        top_k=body.top_k,
        min_similarity=body.min_similarity,
    )
    if result.get("blocked"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Query blocked by Responsible AI Guardrails",
                "guardrail_report": result.get("guardrail_report"),
            },
        )
    return result


@router.get("/rag/documents")
async def list_brand_documents(user_id: str = Depends(require_user)):
    """List brand documents indexed by the authenticated user."""
    rag = get_rag_service()
    docs = await rag.list_documents(user_id=user_id)
    return {"documents": docs, "count": len(docs)}


@router.delete("/rag/documents/{doc_id}")
async def delete_brand_document(doc_id: str, user_id: str = Depends(require_user)):
    """Delete a brand document and its vector chunks for the authenticated user."""
    rag = get_rag_service()
    deleted = await rag.delete_document(user_id=user_id, doc_id=doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "document_id": doc_id}


@router.post("/agent/direct")
async def run_scene_director_agent(
    body: DirectorAgentRequest,
    user_id: str = Depends(require_user),
):
    """
    Execute the multi-step tool-using Scene Director Agent:
      1. Prompt-injection & PII guardrails
      2. Hybrid RAG citation retrieval
      3. Multi-scene visual + narration storyboard planning
      4. Groundedness & pacing self-critique verification
    """
    agent = get_director_agent()
    result = await agent.direct_video(
        user_id=user_id,
        topic=body.topic,
        tone=body.tone,
        target_duration_seconds=body.target_duration_seconds,
        aspect_ratio=body.aspect_ratio,
        use_rag=body.use_rag,
        top_k=body.top_k,
    )
    if result.get("blocked"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": result.get("reason", "Blocked by Responsible AI Guardrails"),
                "guardrail_report": result.get("guardrail_report"),
                "steps": result.get("steps", []),
            },
        )
    return result


@router.post("/agent/evaluate")
async def evaluate_ai_pipeline(
    body: EvaluateRequest = EvaluateRequest(),
    user_id: str = Depends(require_user),
):
    """Run automated RAG, Agent, and Guardrails LLMOps evaluation suite."""
    evaluator = get_ai_evaluator()
    return await evaluator.run_benchmark(
        user_id=user_id,
        persist_results=body.persist_results,
        use_remote_rag=body.use_remote_rag,
    )
