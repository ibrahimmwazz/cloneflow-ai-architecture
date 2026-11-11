"""
CloneFlow AI - Text-to-Video Generation Routes
Powered by CogVideoX-5B on Modal A10G GPU.
Output: 720×480, up to 8 seconds, cinematic quality.
"""

import os
import uuid
import logging
import httpx
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from pydantic import BaseModel, Field

from api.services.supabase_client import get_supabase, SupabaseClient
from api.dependencies.auth import require_user, optional_user

router = APIRouter(prefix="/video-gen", tags=["Video Generation"])
logger = logging.getLogger(__name__)


# ── Request / Response models ───────────────────────────────────────────────

class VideoGenCreateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500,
                        description="Describe the video you want to generate")
    model: str = Field("cogvideox", description="Video model: cogvideox or hunyuan")
    num_frames: int = Field(48, ge=8, le=48,
                            description="Number of frames. 48 = ~6s, 32 = ~4s, 24 = ~3s, 16 = ~2s, 8 = ~1s")
    num_steps: int = Field(30, ge=20, le=50,
                           description="Inference steps. Higher = better quality, slower")
    guidance_scale: float = Field(6.0, ge=1.0, le=15.0,
                                  description="How closely to follow the prompt")
    seed: int = Field(42, description="Seed for reproducibility")


class VideoGenResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    estimated_time_seconds: int = 90
    engine: str = "cogvideox-5b"


_BACKGROUND_TASKS = set()

def _dispatch_background_task(coro):
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


# ── Trigger helper ───────────────────────────────────────────────────────────

async def _trigger_modal_cogvideo(job_id: str, user_id: str,
                                   body: VideoGenCreateRequest):
    """Fire-and-forget background task using independent Supabase client."""
    from api.services.supabase_client import SupabaseClient
    supabase = SupabaseClient()

    if body.model == "hunyuan":
        modal_url = os.getenv("MODAL_HUNYUAN_URL") or "https://ibrahimmustaphawaziri334--cloneflow-worker-web.modal.run/trigger_hunyuan_video"
        logger.info(f"Using HunyuanVideo model for job {job_id[:8]}")
    else:
        modal_url = os.getenv("MODAL_VIDGEN_URL") or "https://ibrahimmustaphawaziri334--cloneflow-worker-web.modal.run/trigger_video_gen"
        logger.info(f"Using CogVideoX model for job {job_id[:8]}")

    if not modal_url:
        error_msg = f"Modal URL not set for model={body.model} — video gen service unavailable"
        logger.error(error_msg)
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(modal_url, json={
                "job_id":         job_id,
                "user_id":        user_id,
                "prompt":         body.prompt,
                "num_frames":     body.num_frames,
                "num_steps":      body.num_steps,
                "guidance_scale": body.guidance_scale,
                "seed":           body.seed,
            })
        if r.status_code == 200:
            logger.info(f"Video gen job {job_id[:8]} dispatched → Modal HTTP {r.status_code}")
            await supabase.update_job_status(job_id, "processing")
            try:
                from api.services.redis_queue import get_redis_queue
                import json
                redis = get_redis_queue()
                redis_client = await redis.client
                if redis_client:
                    await redis_client.publish(f"job_status:{job_id}", json.dumps({
                        "job_id": job_id,
                        "status": "processing",
                        "progress": 15,
                        "timestamp": datetime.utcnow().isoformat()
                    }))
            except Exception as e:
                logger.warning(f"Redis status publish failed: {e}")
        else:
            error_msg = f"Modal returned HTTP {r.status_code}: {r.text[:200]}"
            logger.error(f"Video gen Modal trigger failed for {job_id[:8]}: {error_msg}")
            await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
    except httpx.TimeoutException:
        error_msg = "Modal request timed out (30s) - service may be cold-starting"
        logger.error(f"Video gen Modal timeout for {job_id[:8]}: {error_msg}")
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
    except Exception as e:
        error_msg = f"Modal trigger failed: {str(e)}"
        logger.error(f"Video gen Modal trigger failed for {job_id[:8]}: {error_msg}")
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/create", response_model=VideoGenResponse, status_code=202)
async def create_video_gen(
    body: VideoGenCreateRequest,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Queue a CogVideoX-5B text-to-video job on Modal A10G.
    Returns immediately (202). Poll /api/video-gen/{job_id} for status.
    Completed jobs have video_url pointing to R2.
    """
    job_id = str(uuid.uuid4())

    # Credit check (15 credits per video gen) and deduct upfront
    user_credits = await supabase.get_user_credits(user_id)
    if user_credits < 15:
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits. Text-to-video requires 15 credits."
        )

    deducted = await supabase.deduct_user_credits(user_id, 15)
    if not deducted:
        raise HTTPException(
            status_code=402,
            detail="Credit deduction failed. Please try again."
        )

    # Persist job record
    await supabase.create_job({
        "job_id":             job_id,
        "user_id":            user_id,
        "status":             "pending",
        "input_data":         body.dict(),
        "created_at":         datetime.utcnow().isoformat(),
        "credits_estimated":  15,
        "credits_deducted":   True,
    })
    await supabase.update_job_status(job_id, "queued")

    # Dispatch to Modal in background task with strong reference to prevent GC
    _dispatch_background_task(_trigger_modal_cogvideo(job_id, user_id, body))

    return VideoGenResponse(
        job_id=job_id,
        status="queued",
        created_at=datetime.utcnow(),
        estimated_time_seconds=35,
    )


@router.get("/{job_id}", response_model=VideoGenResponse)
async def get_video_gen_status(
    job_id: str,
    user_id: Optional[str] = Depends(optional_user),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Poll job status. Frontend polls every 3 s until status = completed | failed.
    When completed, video_url contains the R2 CDN URL of the generated MP4.
    """
    if user_id:
        job = await supabase.get_job(job_id, user_id)
    else:
        job = await supabase.get_job_public(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return VideoGenResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=datetime.fromisoformat(job["created_at"]),
        video_url=job.get("video_url"),
        error_message=job.get("error_message"),
        estimated_time_seconds=90,
    )
