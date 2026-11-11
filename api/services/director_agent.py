"""
CloneFlow AI — Multi-Step Tool-Using Scene Director Agent
=========================================================
Executes an explicit tool-using workflow with bounded self-critique:
  1. `guardrail_scan`           — Blocks prompt injection/abuse & redacts PII
  2. `retrieve_brand_knowledge` — Queries tenant pgvector / Hybrid RRF RAG index
  3. `compose_storyboard`       — Synthesizes spoken script + CogVideoX/Hunyuan visual scenes
  4. `verify_and_critique`      — Verifies pacing budget, RAG grounding, & output safety
                                  (triggers at most 1 bounded refinement round on failure)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import httpx

from api.services.ai_guardrails import AIGuardrails
from api.services.rag_service import RAGService, get_rag_service, _tokenize_terms

logger = logging.getLogger(__name__)

WORDS_PER_SECOND = 2.5

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "your", "you", "are",
    "was", "were", "will", "have", "has", "into", "about", "our", "out", "all",
    "can", "how", "why", "what", "when", "where", "who", "its", "more", "than",
    "then", "also", "just", "over", "such", "use", "using", "used", "make",
}


def _extract_salient_terms(texts: List[str], limit: int = 12) -> List[str]:
    counts: Dict[str, int] = {}
    for text in texts:
        for tok in _tokenize_terms(text):
            if len(tok) >= 4 and tok not in _STOPWORDS:
                counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ranked[:limit]]


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Safely extract a JSON object from raw LLM output."""
    if not raw:
        return None
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", cleaned, re.S)
        if brace_match:
            cleaned = brace_match.group(1)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


@dataclass
class AgentStepTrace:
    step: int
    tool: str
    status: str
    duration_ms: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "summary": self.summary,
        }


class SceneDirectorAgent:
    """
    Stateful multi-step agent orchestrating RAG-grounded video direction
    with explicit tool registration, trace telemetry, and self-critique.
    """

    def __init__(self, rag_service: Optional[RAGService] = None):
        self.rag = rag_service or get_rag_service()
        self.tools: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            "guardrail_scan": self._dispatch_guardrail_scan,
            "retrieve_brand_knowledge": self._dispatch_retrieve_brand_knowledge,
            "compose_storyboard": self._dispatch_compose_storyboard,
            "verify_and_critique": self._dispatch_verify_and_critique,
        }

    async def _dispatch_guardrail_scan(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return await self._tool_guardrail_scan(*args, **kwargs)

    async def _dispatch_retrieve_brand_knowledge(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return await self._tool_retrieve_brand_knowledge(*args, **kwargs)

    async def _dispatch_compose_storyboard(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return await self._tool_compose_storyboard(*args, **kwargs)

    async def _dispatch_verify_and_critique(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return await self._tool_verify_and_critique(*args, **kwargs)

    async def _tool_guardrail_scan(self, brief: str) -> Dict[str, Any]:
        report = AIGuardrails.inspect_input(brief, redact_pii=True, max_chars=4000)
        return {
            "allowed": report.allowed,
            "sanitized_text": report.sanitized_text,
            "report": report.to_dict(),
        }

    async def _tool_retrieve_brand_knowledge(
        self,
        user_id: str,
        query: str,
        top_k: int = 4,
        use_remote_db: bool = True,
    ) -> Dict[str, Any]:
        res = await self.rag.query(
            user_id=user_id,
            query_text=query,
            top_k=top_k,
            min_similarity=0.05,
            use_remote_db=use_remote_db,
        )
        chunks = res.get("chunks", [])
        citations = [
            {
                "citation_id": f"[{idx + 1}]",
                "document_id": c["document_id"],
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "chunk_index": c["chunk_index"],
                "vector_score": c["vector_score"],
                "lexical_score": c["lexical_score"],
                "rrf_score": c["rrf_score"],
                "snippet": c["content"][:220],
            }
            for idx, c in enumerate(chunks)
        ]
        return {
            "chunks": chunks,
            "citations": citations,
            "retrieval_backend": res.get("retrieval_backend", "memory"),
            "embedding_provider": res.get("embedding_provider", "unknown"),
        }

    def _recommended_model(self, aspect_ratio: str, num_scenes: int) -> str:
        if aspect_ratio.strip() in {"9:16", "4:5", "vertical", "portrait"}:
            return "hunyuan"
        if num_scenes <= 1:
            return "cogvideox"
        return "cogvideox"

    def _grounded_deterministic_storyboard(
        self,
        topic: str,
        tone: str,
        target_duration: int,
        aspect_ratio: str,
        chunks: List[Dict[str, Any]],
        critique_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize a grounded, pace-calibrated storyboard cleanly when external LLM APIs
        are unavailable or fail schema validation.
        """
        num_scenes = max(1, min(6, round(target_duration / 10)))
        scene_dur = max(5, round(target_duration / num_scenes))
        target_words = max(10, int(round(target_duration * WORDS_PER_SECOND)))

        tone_styles = {
            "professional": "confident executive keynote style, crisp lighting, clean studio framing",
            "casual": "warm approachable creator style, natural daylight, lifestyle setting",
            "energetic": "dynamic kinetic transitions, high-contrast studio lighting, punchy delivery",
            "educational": "structured visual explainer style, clean infographic composition",
        }
        visual_style = tone_styles.get(tone, tone_styles["professional"])
        motions = ["slow dolly-in", "smooth lateral pan", "subtle push-in", "steady medium close-up"]

        # Extract concrete sentences from retrieved RAG chunks so narration preserves grounded facts
        fact_sentences: List[str] = []
        for chunk in chunks:
            for sent in re.split(r"(?<=[.!?])\s+", chunk.get("content", "").strip()):
                cleaned_sent = sent.strip()
                if len(cleaned_sent.split()) >= 4 and cleaned_sent not in fact_sentences:
                    fact_sentences.append(cleaned_sent)

        opener_map = {
            "professional": f"Welcome. Today we examine {topic}.",
            "casual": f"Hey there! Let's break down {topic}.",
            "energetic": f"Here is why {topic} is changing everything right now!",
            "educational": f"In this brief walkthrough, let's explore {topic}.",
        }
        closer_map = {
            "professional": f"Adopt {topic} today to accelerate reliable production outcomes.",
            "casual": f"Give {topic} a try and see the workflow difference firsthand.",
            "energetic": f"Put {topic} into action right now and scale your creative velocity!",
            "educational": f"These grounded fundamentals summarize what makes {topic} effective.",
        }

        narrations: List[str] = []
        if fact_sentences:
            body_pool = fact_sentences[: max(2, num_scenes + 1)]
        else:
            body_pool = [
                f"{topic} streamlines creation from initial concept to studio-grade video delivery.",
                f"Every scene aligns spoken clarity with cohesive visual direction for {topic}.",
                f"Consistent pacing keeps viewers engaged across the entire {target_duration}-second sequence.",
            ]

        scene_connectors = [
            f"Each stage highlights how {topic} translates complex goals into crisp execution.",
            f"Teams rely on {topic} to maintain visual continuity and voice clarity across channels.",
            f"From opening hook to final call to action, {topic} keeps messaging focused and measurable.",
            f"Built for real-world deployment, {topic} adapts effortlessly across desktop and mobile formats.",
        ]

        words_per_scene = max(8, target_words // num_scenes)
        for idx in range(num_scenes):
            parts: List[str] = []
            if idx == 0:
                parts.append(opener_map.get(tone, opener_map["professional"]))
            fact = body_pool[idx % len(body_pool)]
            parts.append(fact)

            seg_text = " ".join(parts).strip()
            connector_idx = idx
            while len(seg_text.split()) < words_per_scene - 2:
                extra = scene_connectors[connector_idx % len(scene_connectors)]
                if extra in seg_text and len(body_pool) > 1:
                    alt = body_pool[(idx + 1) % len(body_pool)]
                    if alt not in seg_text:
                        seg_text = f"{seg_text} {alt}"
                        connector_idx += 1
                        continue
                seg_text = f"{seg_text} {extra}"
                connector_idx += 1
                if connector_idx - idx >= 2:
                    break

            if idx == num_scenes - 1:
                seg_text = f"{seg_text} {closer_map.get(tone, closer_map['professional'])}".strip()

            seg_words = seg_text.split()
            if critique_hint == "trim_length" and len(seg_words) > words_per_scene + 4:
                seg_text = " ".join(seg_words[: words_per_scene + 2]).rstrip(",;:") + "."
            narrations.append(seg_text)

        scenes = []
        for idx in range(num_scenes):
            motion = motions[idx % len(motions)]
            focus_snippet = fact_sentences[idx % len(fact_sentences)][:90] if fact_sentences else topic
            scenes.append({
                "scene_index": idx + 1,
                "duration_seconds": scene_dur,
                "camera_motion": motion,
                "visual_prompt": (
                    f"Cinematic {aspect_ratio} shot ({visual_style}): illustrating {topic} — "
                    f"visualizing '{focus_snippet}', {motion}, ultra-detailed 4k film grain."
                ),
                "narration": narrations[idx],
            })

        full_script = " ".join(s["narration"] for s in scenes).strip()
        # Trim or pad gently if needed to keep exact duration pacing within +/-20%
        words = full_script.split()
        max_allowed = int(target_words * 1.25)
        if len(words) > max_allowed:
            full_script = " ".join(words[:max_allowed]).rstrip(",;:") + "."

        return {
            "recommended_model": self._recommended_model(aspect_ratio, num_scenes),
            "scenes": scenes,
            "full_script": full_script,
            "provider": "grounded-deterministic-planner",
        }

    async def _tool_compose_storyboard(
        self,
        topic: str,
        tone: str,
        target_duration: int,
        aspect_ratio: str,
        chunks: List[Dict[str, Any]],
        critique_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        num_scenes = max(1, min(6, round(target_duration / 10)))
        target_words = max(10, int(round(target_duration * WORDS_PER_SECOND)))

        context_block = "\n".join(
            f"[{i + 1}] ({c.get('title', 'Doc')}): {c['content']}"
            for i, c in enumerate(chunks)
        )
        system_prompt = (
            "You are CloneFlow AI's Senior Video Director Agent. "
            "Respond ONLY with valid JSON matching schema: "
            '{"recommended_model":"cogvideox|hunyuan","scenes":[{"scene_index":1,'
            '"duration_seconds":10,"camera_motion":"...","visual_prompt":"...",'
            '"narration":"..."}],"full_script":"..."}. '
            "Ground factual claims strictly in the provided Brand Knowledge Context when present. "
            "Do NOT include stage directions in full_script."
        )
        user_prompt = (
            f"Topic: {topic}\n"
            f"Tone: {tone}\n"
            f"Target Duration: {target_duration}s (~{target_words} words total across {num_scenes} scenes)\n"
            f"Aspect Ratio: {aspect_ratio}\n"
        )
        if critique_hint:
            user_prompt += f"Revision Requirement: {critique_hint}\n"
        if context_block:
            user_prompt += f"\nBrand Knowledge Context:\n{context_block}\n"

        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                async with httpx.AsyncClient(timeout=20) as http:
                    resp = await http.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.4,
                            "max_tokens": 1100,
                        },
                    )
                    if resp.is_success:
                        raw = resp.json()["choices"][0]["message"]["content"]
                        parsed = _extract_json_object(raw)
                        if parsed and isinstance(parsed.get("scenes"), list) and parsed.get("scenes"):
                            full_script = str(parsed.get("full_script") or " ".join(
                                str(s.get("narration", "")) for s in parsed["scenes"]
                            )).strip()
                            if full_script:
                                parsed["full_script"] = full_script
                                parsed.setdefault("recommended_model", self._recommended_model(aspect_ratio, num_scenes))
                                parsed["provider"] = "groq:llama-3.1-8b-instant"
                                return parsed
            except Exception as e:
                logger.warning(f"Director Agent Groq fallback triggered: {e}")

        hf_token = os.getenv("HF_TOKEN", "").strip()
        if hf_token:
            try:
                async with httpx.AsyncClient(timeout=20) as http:
                    resp = await http.post(
                        "https://router.huggingface.co/v1/chat/completions",
                        headers={"Authorization": f"Bearer {hf_token}"},
                        json={
                            "model": "Qwen/Qwen2.5-7B-Instruct",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.4,
                            "max_tokens": 1100,
                        },
                    )
                    if resp.is_success:
                        raw = resp.json()["choices"][0]["message"]["content"]
                        parsed = _extract_json_object(raw)
                        if parsed and isinstance(parsed.get("scenes"), list) and parsed.get("scenes"):
                            full_script = str(parsed.get("full_script") or " ".join(
                                str(s.get("narration", "")) for s in parsed["scenes"]
                            )).strip()
                            if full_script:
                                parsed["full_script"] = full_script
                                parsed.setdefault("recommended_model", self._recommended_model(aspect_ratio, num_scenes))
                                parsed["provider"] = "huggingface:Qwen/Qwen2.5-7B-Instruct"
                                return parsed
            except Exception as e:
                logger.warning(f"Director Agent HF Router fallback triggered: {e}")

        return self._grounded_deterministic_storyboard(
            topic=topic,
            tone=tone,
            target_duration=target_duration,
            aspect_ratio=aspect_ratio,
            chunks=chunks,
            critique_hint=critique_hint,
        )

    async def _tool_verify_and_critique(
        self,
        plan: Dict[str, Any],
        target_duration: int,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        full_script = str(plan.get("full_script", "")).strip()
        word_count = len(full_script.split())
        target_words = max(8, int(round(target_duration * WORDS_PER_SECOND)))
        pacing_error_ratio = abs(word_count - target_words) / max(1, target_words)
        pacing_ok = pacing_error_ratio <= 0.45

        combined_visuals = " ".join(str(s.get("visual_prompt", "")) for s in plan.get("scenes", []))
        out_guard = AIGuardrails.validate_output(f"{full_script} {combined_visuals}")

        grounding_score = 1.0
        if chunks:
            top_sim = float(chunks[0].get("vector_score", 0.0))
            primary_chunks = [
                c["content"] for c in chunks
                if float(c.get("vector_score", 0.0)) >= max(0.08, top_sim * 0.45)
            ] or [chunks[0]["content"]]
            salient_terms = _extract_salient_terms(primary_chunks, limit=10)
            script_terms = set(_tokenize_terms(f"{full_script} {combined_visuals}"))
            if salient_terms:
                matched = sum(1 for t in salient_terms if t in script_terms)
                grounding_score = matched / len(salient_terms)
        grounding_ok = (not chunks) or (grounding_score >= 0.25)

        issues: List[str] = []
        critique_hint: Optional[str] = None
        if not out_guard.allowed:
            issues.append(f"output_guardrail:{out_guard.flags}")
        if not pacing_ok:
            issues.append(f"pacing_mismatch:{word_count}_vs_{target_words}")
            critique_hint = "trim_length" if word_count > target_words else "expand_detail"
        if not grounding_ok:
            issues.append(f"low_rag_grounding:{grounding_score:.2f}")
            critique_hint = "Preserve exact facts from Brand Knowledge Context."

        return {
            "passed": out_guard.allowed and pacing_ok and grounding_ok,
            "word_count": word_count,
            "target_words": target_words,
            "pacing_error_ratio": round(pacing_error_ratio, 4),
            "grounding_score": round(grounding_score, 4),
            "output_guardrail": out_guard.to_dict(),
            "issues": issues,
            "critique_hint": critique_hint,
        }

    async def direct_video(
        self,
        user_id: str,
        topic: str,
        tone: str = "professional",
        target_duration_seconds: int = 30,
        aspect_ratio: str = "16:9",
        use_rag: bool = True,
        top_k: int = 4,
        use_remote_db: bool = True,
    ) -> Dict[str, Any]:
        """Run the full multi-step tool-using Scene Director Agent loop."""
        t0 = time.perf_counter()
        steps: List[AgentStepTrace] = []
        tools_invoked: List[str] = []

        # Step 1: Input Guardrails
        s1 = time.perf_counter()
        g_res = await self.tools["guardrail_scan"](topic)
        tools_invoked.append("guardrail_scan")
        steps.append(AgentStepTrace(
            step=1,
            tool="guardrail_scan",
            status="passed" if g_res["allowed"] else "blocked",
            duration_ms=(time.perf_counter() - s1) * 1000,
            summary=f"allowed={g_res['allowed']} flags={g_res['report']['flags']}",
        ))

        if not g_res["allowed"]:
            return {
                "blocked": True,
                "reason": "Request blocked by Responsible AI Guardrails",
                "guardrail_report": g_res["report"],
                "recommended_model": "none",
                "full_script": "",
                "scenes": [],
                "citations": [],
                "grounding_score": 0.0,
                "pacing_error_ratio": 1.0,
                "refinement_rounds": 0,
                "verification_passed": False,
                "llm_provider": "blocked",
                "tools_invoked": tools_invoked,
                "steps": [st.to_dict() for st in steps],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        sanitized_topic = g_res["sanitized_text"]

        # Step 2: Brand RAG Retrieval (when enabled)
        chunks: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        retrieval_backend = "disabled"
        embedding_provider = "disabled"

        if use_rag:
            s2 = time.perf_counter()
            rag_res = await self.tools["retrieve_brand_knowledge"](
                user_id=user_id,
                query=sanitized_topic,
                top_k=top_k,
                use_remote_db=use_remote_db,
            )
            chunks = rag_res["chunks"]
            citations = rag_res["citations"]
            retrieval_backend = rag_res["retrieval_backend"]
            embedding_provider = rag_res["embedding_provider"]
            tools_invoked.append("retrieve_brand_knowledge")
            steps.append(AgentStepTrace(
                step=len(steps) + 1,
                tool="retrieve_brand_knowledge",
                status="completed",
                duration_ms=(time.perf_counter() - s2) * 1000,
                summary=f"retrieved {len(chunks)} chunks via {retrieval_backend} ({embedding_provider})",
            ))

        # Step 3: Initial Storyboard & Visual Prompt Composition
        s3 = time.perf_counter()
        plan = await self.tools["compose_storyboard"](
            topic=sanitized_topic,
            tone=tone,
            target_duration=target_duration_seconds,
            aspect_ratio=aspect_ratio,
            chunks=chunks,
            critique_hint=None,
        )
        tools_invoked.append("compose_storyboard")
        steps.append(AgentStepTrace(
            step=len(steps) + 1,
            tool="compose_storyboard",
            status="completed",
            duration_ms=(time.perf_counter() - s3) * 1000,
            summary=f"{len(plan['scenes'])} scenes via {plan['provider']}",
        ))

        # Step 4: Self-Critique Verification (+ max 1 bounded refinement pass)
        s4 = time.perf_counter()
        critique = await self.tools["verify_and_critique"](
            plan=plan,
            target_duration=target_duration_seconds,
            chunks=chunks,
        )
        tools_invoked.append("verify_and_critique")
        steps.append(AgentStepTrace(
            step=len(steps) + 1,
            tool="verify_and_critique",
            status="passed" if critique["passed"] else "needs_refinement",
            duration_ms=(time.perf_counter() - s4) * 1000,
            summary=f"pacing_err={critique['pacing_error_ratio']} grounding={critique['grounding_score']}",
        ))

        refinement_rounds = 0
        if not critique["passed"] and critique.get("critique_hint"):
            refinement_rounds = 1
            s5 = time.perf_counter()
            refined = await self._tool_compose_storyboard(
                topic=sanitized_topic,
                tone=tone,
                target_duration=target_duration_seconds,
                aspect_ratio=aspect_ratio,
                chunks=chunks,
                critique_hint=critique["critique_hint"],
            )
            # If LLM refinement returned weak grounding/pacing, fall back to calibrated deterministic synthesis
            refined_critique = await self._tool_verify_and_critique(
                plan=refined,
                target_duration=target_duration_seconds,
                chunks=chunks,
            )
            if not refined_critique["passed"] and refined.get("provider") != "grounded-deterministic-planner":
                refined = self._grounded_deterministic_storyboard(
                    topic=sanitized_topic,
                    tone=tone,
                    target_duration=target_duration_seconds,
                    aspect_ratio=aspect_ratio,
                    chunks=chunks,
                    critique_hint=critique["critique_hint"],
                )
                refined_critique = await self._tool_verify_and_critique(
                    plan=refined,
                    target_duration=target_duration_seconds,
                    chunks=chunks,
                )
            plan = refined
            critique = refined_critique
            tools_invoked.extend(["compose_storyboard", "verify_and_critique"])
            steps.append(AgentStepTrace(
                step=len(steps) + 1,
                tool="refine_and_reverify",
                status="passed" if critique["passed"] else "warning",
                duration_ms=(time.perf_counter() - s5) * 1000,
                summary=f"round=1 passed={critique['passed']} grounding={critique['grounding_score']}",
            ))

        return {
            "blocked": False,
            "topic": sanitized_topic,
            "tone": tone,
            "target_duration_seconds": target_duration_seconds,
            "aspect_ratio": aspect_ratio,
            "recommended_model": plan["recommended_model"],
            "full_script": plan["full_script"],
            "word_count": critique["word_count"],
            "scenes": plan["scenes"],
            "scene_prompts": [s["visual_prompt"] for s in plan["scenes"]],
            "citations": citations,
            "retrieval_backend": retrieval_backend,
            "embedding_provider": embedding_provider,
            "guardrail_report": g_res["report"],
            "grounding_score": critique["grounding_score"],
            "pacing_error_ratio": critique["pacing_error_ratio"],
            "refinement_rounds": refinement_rounds,
            "verification_passed": critique["passed"],
            "llm_provider": plan["provider"],
            "tools_invoked": tools_invoked,
            "steps": [st.to_dict() for st in steps],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


_director_singleton: Optional[SceneDirectorAgent] = None


def get_director_agent() -> SceneDirectorAgent:
    global _director_singleton
    if _director_singleton is None:
        _director_singleton = SceneDirectorAgent()
    return _director_singleton
