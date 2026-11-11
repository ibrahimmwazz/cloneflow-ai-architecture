"""
CloneFlow AI — Hybrid RAG & Vector Embedding Service
=====================================================
Provides:
  1. Sentence-aware sliding-window chunking with SHA-256 deduplication
  2. Multi-provider 384-d embeddings:
       - Hugging Face Inference API (BAAI/bge-small-en-v1.5) when HF_TOKEN is configured
       - OpenAI Embeddings API (text-embedding-3-small, 384-d) when OPENAI_API_KEY is configured
       - Deterministic 384-d L2-normalized n-gram hash projection fallback for offline/CI runs
  3. Tenant-isolated vector storage via Supabase pgvector (`match_brand_chunks` RPC)
     with process-local fallback store for zero-dependency test execution
  4. Hybrid search combining Dense Cosine Similarity + Lexical Overlap via
     Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from api.services.ai_guardrails import AIGuardrails

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return [0.0] * len(vec)
    return [v / norm for v in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def _stem_token(tok: str) -> str:
    if len(tok) <= 4:
        return tok
    for suffix in (
        "ifications", "ification", "ations", "ation", "ments", "ment",
        "ings", "ing", "iers", "ies", "ied", "ers", "er", "ions", "ion",
        "ed", "es", "s",
    ):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 4:
            return tok[: -len(suffix)]
    return tok


def _tokenize_terms(text: str) -> List[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem_token(t) for t in raw]


def chunk_text(
    text: str,
    chunk_size_chars: int = 480,
    overlap_chars: int = 90,
) -> List[str]:
    """
    Sentence-aware sliding-window chunker with overlap.
    Preserves sentence boundaries where possible so semantic meaning stays intact.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size_chars:
        return [cleaned]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > chunk_size_chars:
            # Split extra-long single sentences cleanly by words
            words = sentence.split()
            for w in words:
                candidate = f"{current} {w}".strip()
                if len(candidate) <= chunk_size_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                        tail = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
                        current = f"{tail} {w}".strip() if tail else w
                    else:
                        chunks.append(w[:chunk_size_chars])
                        current = ""
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
                tail = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
                # Start next chunk on clean boundary using overlap tail if fits
                current = f"{tail} {sentence}".strip() if tail and len(tail) + len(sentence) + 1 <= chunk_size_chars else sentence
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


class EmbeddingEngine:
    """Computes L2-normalized 384-d embeddings with Hugging Face, OpenAI, or local hash fallback."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    def _local_hash_embedding(self, text: str) -> List[float]:
        """
        Deterministic 384-d signed feature-hashing projection over unigrams + bigrams.
        Enables zero-cost local similarity matching and deterministic CI evaluation.
        """
        vec = [0.0] * EMBEDDING_DIM
        tokens = _tokenize_terms(text)
        if not tokens:
            return vec

        features: List[Tuple[str, float]] = [(t, 1.2) for t in tokens]
        features.extend((f"{tokens[i]}_{tokens[i+1]}", 1.5) for i in range(len(tokens) - 1))
        for tok in tokens:
            if len(tok) >= 4:
                for i in range(len(tok) - 3):
                    features.append((f"c4:{tok[i:i+4]}", 0.45))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[bucket] += sign * weight

        return _l2_normalize(vec)

    async def embed(self, text: str) -> Tuple[List[float], str]:
        """Return (384-d normalized embedding, provider_label)."""
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            try:
                url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}/pipeline/feature-extraction"
                async with httpx.AsyncClient(timeout=12) as client:
                    resp = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {hf_token}"},
                        json={"inputs": text[:2000], "options": {"wait_for_model": True}},
                    )
                    if resp.is_success:
                        data = resp.json()
                        # Handle [384] or [[token_vecs...]] response shapes
                        if isinstance(data, list) and data:
                            if isinstance(data[0], list):
                                # Mean-pool token vectors
                                dims = len(data[0])
                                pooled = [
                                    sum(row[i] for row in data) / len(data)
                                    for i in range(min(dims, EMBEDDING_DIM))
                                ]
                                if len(pooled) == EMBEDDING_DIM:
                                    return _l2_normalize(pooled), f"huggingface:{self.model_name}"
                            elif len(data) == EMBEDDING_DIM:
                                return _l2_normalize([float(x) for x in data]), f"huggingface:{self.model_name}"
            except Exception as e:
                logger.warning(f"HF embedding API fallback triggered: {e}")

        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                async with httpx.AsyncClient(timeout=12) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        json={
                            "model": "text-embedding-3-small",
                            "input": text[:4000],
                            "dimensions": EMBEDDING_DIM,
                        },
                    )
                    if resp.is_success:
                        emb = resp.json()["data"][0]["embedding"]
                        if len(emb) == EMBEDDING_DIM:
                            return _l2_normalize([float(x) for x in emb]), "openai:text-embedding-3-small"
            except Exception as e:
                logger.warning(f"OpenAI embedding API fallback triggered: {e}")

        return self._local_hash_embedding(text), "local-hash-384"


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    chunk_index: int
    content: str
    vector_score: float
    lexical_score: float
    rrf_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "vector_score": round(self.vector_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "rrf_score": round(self.rrf_score, 4),
            "metadata": self.metadata,
        }


class RAGService:
    """
    Production Hybrid RAG service backed by Supabase pgvector when configured,
    with an isolated in-memory vector store for local dev / offline testing.
    """

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.embedder = embedding_engine or EmbeddingEngine()
        # Process-local store keyed by user_id for unit tests / offline execution
        self._local_docs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._local_chunks: Dict[str, List[Dict[str, Any]]] = {}

    def _supabase_config(self) -> Optional[Tuple[str, str]]:
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if url.startswith("https://") and key and "test.supabase.co" not in url:
            return url.rstrip("/"), key
        return None

    async def ingest_document(
        self,
        user_id: str,
        title: str,
        content: str,
        source_type: str = "brand_guide",
        metadata: Optional[Dict[str, Any]] = None,
        use_remote_db: bool = True,
    ) -> Dict[str, Any]:
        """Inspect, sanitize, chunk, embed, and persist a brand/knowledge document."""
        guard = AIGuardrails.inspect_input(content, redact_pii=True, max_chars=24000)
        if not guard.allowed:
            raise ValueError(f"Document rejected by AI guardrails: {guard.flags}")

        clean_title_guard = AIGuardrails.inspect_input(title, redact_pii=True, max_chars=200)
        clean_title = clean_title_guard.sanitized_text if clean_title_guard.allowed else "Untitled Document"

        chunks = chunk_text(guard.sanitized_text)
        if not chunks:
            raise ValueError("Document produced no indexable chunks")

        content_hash = hashlib.sha256(guard.sanitized_text.encode("utf-8")).hexdigest()
        doc_id = str(uuid.uuid4())
        doc_meta = {**(metadata or {}), "title": clean_title, "source_type": source_type}

        embedded_chunks: List[Dict[str, Any]] = []
        provider_used = "local-hash-384"

        for idx, text_chunk in enumerate(chunks):
            vec, provider = await self.embedder.embed(text_chunk)
            provider_used = provider
            embedded_chunks.append({
                "id": str(uuid.uuid4()),
                "document_id": doc_id,
                "user_id": user_id,
                "chunk_index": idx,
                "content": text_chunk,
                "token_estimate": max(1, int(len(text_chunk.split()) * 1.3)),
                "metadata": {**doc_meta, "embedding_provider": provider},
                "embedding": vec,
            })

        stored_backend = "memory"
        sb = self._supabase_config() if use_remote_db else None
        if sb:
            base_url, service_key = sb
            headers = {
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
            }
            try:
                async with httpx.AsyncClient(timeout=15) as http:
                    doc_resp = await http.post(
                        f"{base_url}/rest/v1/brand_documents",
                        headers={**headers, "Prefer": "return=representation"},
                        json={
                            "id": doc_id,
                            "user_id": user_id,
                            "title": clean_title,
                            "source_type": source_type,
                            "content_hash": content_hash,
                            "chunk_count": len(embedded_chunks),
                            "metadata": doc_meta,
                        },
                    )
                    if doc_resp.is_success:
                        rows = [
                            {
                                "id": c["id"],
                                "document_id": doc_id,
                                "user_id": user_id,
                                "chunk_index": c["chunk_index"],
                                "content": c["content"],
                                "token_estimate": c["token_estimate"],
                                "metadata": c["metadata"],
                                # PostgREST expects pgvector literal string '[0.1,0.2,...]'
                                "embedding": "[" + ",".join(f"{v:.6f}" for v in c["embedding"]) + "]",
                            }
                            for c in embedded_chunks
                        ]
                        chunk_resp = await http.post(
                            f"{base_url}/rest/v1/brand_chunks",
                            headers=headers,
                            json=rows,
                        )
                        if chunk_resp.is_success:
                            stored_backend = "supabase_pgvector"
                        else:
                            logger.warning(f"pgvector chunk insert failed ({chunk_resp.status_code}); using local mirror")
                    elif doc_resp.status_code == 409:
                        existing_resp = await http.get(
                            f"{base_url}/rest/v1/brand_documents",
                            headers=headers,
                            params={
                                "user_id": f"eq.{user_id}",
                                "content_hash": f"eq.{content_hash}",
                                "select": "id,title,chunk_count",
                            },
                        )
                        if existing_resp.is_success and existing_resp.json():
                            doc_id = existing_resp.json()[0]["id"]
                            stored_backend = "supabase_pgvector_cached"
                    else:
                        logger.warning(f"pgvector doc insert skipped ({doc_resp.status_code}); using local mirror")
            except Exception as e:
                logger.warning(f"Supabase pgvector ingest unreachable, using local store: {e}")

        # Always keep tenant-isolated local store up to date when remote DB isn't active or mirrors
        self._local_docs.setdefault(user_id, {})[doc_id] = {
            "id": doc_id,
            "user_id": user_id,
            "title": clean_title,
            "source_type": source_type,
            "content_hash": content_hash,
            "chunk_count": len(embedded_chunks),
            "metadata": doc_meta,
        }
        existing = [c for c in self._local_chunks.get(user_id, []) if c.get("document_id") != doc_id]
        existing.extend(embedded_chunks)
        self._local_chunks[user_id] = existing

        return {
            "document_id": doc_id,
            "title": clean_title,
            "chunk_count": len(embedded_chunks),
            "embedding_provider": provider_used,
            "storage_backend": stored_backend,
            "guardrail_report": guard.to_dict(),
        }

    async def list_documents(self, user_id: str, use_remote_db: bool = True) -> List[Dict[str, Any]]:
        sb = self._supabase_config() if use_remote_db else None
        if sb:
            base_url, service_key = sb
            try:
                async with httpx.AsyncClient(timeout=10) as http:
                    resp = await http.get(
                        f"{base_url}/rest/v1/brand_documents",
                        headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
                        params={"user_id": f"eq.{user_id}", "order": "created_at.desc"},
                    )
                    if resp.is_success:
                        return resp.json()
            except Exception as e:
                logger.warning(f"Supabase list_documents fallback: {e}")

        return list(self._local_docs.get(user_id, {}).values())

    async def delete_document(self, user_id: str, doc_id: str, use_remote_db: bool = True) -> bool:
        deleted = False
        if doc_id in self._local_docs.get(user_id, {}):
            del self._local_docs[user_id][doc_id]
            self._local_chunks[user_id] = [
                c for c in self._local_chunks.get(user_id, []) if c.get("document_id") != doc_id
            ]
            deleted = True

        sb = self._supabase_config() if use_remote_db else None
        if sb:
            base_url, service_key = sb
            try:
                async with httpx.AsyncClient(timeout=10) as http:
                    resp = await http.delete(
                        f"{base_url}/rest/v1/brand_documents",
                        headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
                        params={"id": f"eq.{doc_id}", "user_id": f"eq.{user_id}"},
                    )
                    if resp.is_success:
                        deleted = True
            except Exception as e:
                logger.warning(f"Supabase delete_document fallback: {e}")

        return deleted

    async def query(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 4,
        min_similarity: float = 0.05,
        use_remote_db: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Hybrid Search (Dense Vector Cosine + Lexical Overlap) with Reciprocal Rank Fusion (RRF).
        """
        guard = AIGuardrails.inspect_input(query_text, redact_pii=True, max_chars=1500)
        if not guard.allowed:
            return {
                "blocked": True,
                "guardrail_report": guard.to_dict(),
                "chunks": [],
                "embedding_provider": "none",
            }

        q_vec, provider = await self.embedder.embed(guard.sanitized_text)
        candidates: List[Dict[str, Any]] = []
        retrieval_backend = "memory"

        sb = self._supabase_config() if use_remote_db else None
        if sb:
            base_url, service_key = sb
            try:
                async with httpx.AsyncClient(timeout=12) as http:
                    rpc_resp = await http.post(
                        f"{base_url}/rest/v1/rpc/match_brand_chunks",
                        headers={
                            "apikey": service_key,
                            "Authorization": f"Bearer {service_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "query_embedding": "[" + ",".join(f"{v:.6f}" for v in q_vec) + "]",
                            "p_user_id": user_id,
                            "match_threshold": min_similarity,
                            "match_count": max(top_k * 3, 10),
                        },
                    )
                    if rpc_resp.is_success and rpc_resp.json():
                        retrieval_backend = "supabase_pgvector"
                        for row in rpc_resp.json():
                            candidates.append({
                                "id": row["id"],
                                "document_id": row["document_id"],
                                "chunk_index": row["chunk_index"],
                                "content": row["content"],
                                "metadata": row.get("metadata") or {},
                                "vector_score": float(row.get("similarity", 0.0)),
                            })
            except Exception as e:
                logger.warning(f"pgvector RPC query fallback to local store: {e}")

        if not candidates:
            for c in self._local_chunks.get(user_id, []):
                sim = cosine_similarity(q_vec, c["embedding"])
                candidates.append({
                    **c,
                    "vector_score": sim,
                })

        if not candidates:
            return {
                "blocked": False,
                "guardrail_report": guard.to_dict(),
                "chunks": [],
                "embedding_provider": provider,
                "retrieval_backend": retrieval_backend,
            }

        # Compute lexical keyword match ratio
        q_terms = set(_tokenize_terms(guard.sanitized_text))
        for item in candidates:
            doc_terms = set(_tokenize_terms(item["content"]))
            overlap = len(q_terms & doc_terms) / max(1, len(q_terms))
            item["lexical_score"] = overlap

        # Hybrid RRF (Reciprocal Rank Fusion, k=60)
        by_vector = sorted(candidates, key=lambda x: x["vector_score"], reverse=True)
        by_lexical = sorted(candidates, key=lambda x: x["lexical_score"], reverse=True)

        rrf_scores: Dict[str, float] = {}
        k_const = 60.0
        for rank, item in enumerate(by_vector, start=1):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (k_const + rank)
        for rank, item in enumerate(by_lexical, start=1):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (k_const + rank)

        filtered = [
            item for item in candidates
            if item["vector_score"] >= min_similarity or item["lexical_score"] > 0.15
        ]
        filtered.sort(key=lambda x: (rrf_scores[x["id"]], x["vector_score"]), reverse=True)

        results = [
            RetrievedChunk(
                chunk_id=item["id"],
                document_id=item["document_id"],
                title=(item.get("metadata") or {}).get("title", "Brand Document"),
                chunk_index=item["chunk_index"],
                content=item["content"],
                vector_score=item["vector_score"],
                lexical_score=item["lexical_score"],
                rrf_score=rrf_scores[item["id"]],
                metadata=item.get("metadata") or {},
            ).to_dict()
            for item in filtered[:top_k]
        ]

        return {
            "blocked": False,
            "guardrail_report": guard.to_dict(),
            "chunks": results,
            "embedding_provider": provider,
            "retrieval_backend": retrieval_backend,
        }


_rag_singleton: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RAGService()
    return _rag_singleton
