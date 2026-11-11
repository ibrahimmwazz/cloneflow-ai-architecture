"""
CloneFlow AI - Image Generation Routes
Powered by Flux.1-schnell on Modal A10G GPU.
Output: 1024×1024, photorealistic quality.
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
from api.dependencies.auth import require_user
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/image-gen", tags=["Image Generation"])
logger = logging.getLogger(__name__)


# ── Request / Response models ───────────────────────────────────────────────

class ImageGenCreateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500,
                        description="Describe the image you want to generate")
    aspect_ratio: str = Field("1:1", description="Aspect ratio: 1:1, 16:9, or 9:16")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    num_inference_steps: int = Field(4, ge=1, le=8,
                                      description="Inference steps. Higher = better quality, slower")
    guidance_scale: float = Field(0.0, ge=0.0, le=10.0,
                                   description="Guidance scale for Flux")
    seed: int = Field(42, description="Seed for reproducibility")


class ImageGenResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    image_url: Optional[str] = None
    error_message: Optional[str] = None
    estimated_time_seconds: int = 10
    engine: str = "flux.1-schnell"


_BACKGROUND_TASKS = set()

def _dispatch_background_task(coro):
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


# ── Trigger helper ───────────────────────────────────────────────────────────

async def _trigger_modal_image_gen(job_id: str, user_id: str,
                                    body: ImageGenCreateRequest):
    """Fire-and-forget background task using independent Supabase client."""
    from api.services.supabase_client import SupabaseClient
    supabase = SupabaseClient()

    modal_url = os.getenv("MODAL_IMAGEGEN_URL") or "https://ibrahimmustaphawaziri334--cloneflow-worker-web.modal.run/trigger_image_gen"
    if not modal_url:
        error_msg = "MODAL_IMAGEGEN_URL not set — image gen service unavailable"
        logger.error(error_msg)
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(modal_url, json={
                "job_id": job_id,
                "user_id": user_id,
                "prompt": body.prompt,
                "aspect_ratio": body.aspect_ratio,
                "num_images": body.num_images,
                "num_inference_steps": body.num_inference_steps,
                "guidance_scale": body.guidance_scale,
                "seed": body.seed,
            })
        if r.status_code == 200:
            logger.info(f"Image gen job {job_id[:8]} dispatched → Modal HTTP {r.status_code}")
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
                        "progress": 25,
                        "timestamp": datetime.utcnow().isoformat()
                    }))
            except Exception as e:
                logger.warning(f"Redis status publish failed: {e}")
        else:
            error_msg = f"Modal returned HTTP {r.status_code}: {r.text[:200]}"
            logger.error(f"Image gen Modal trigger failed for {job_id[:8]}: {error_msg}")
            await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
    except httpx.TimeoutException:
        error_msg = "Modal request timed out (30s) - service may be cold-starting"
        logger.error(f"Image gen Modal timeout for {job_id[:8]}: {error_msg}")
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
    except Exception as e:
        error_msg = f"Modal trigger failed: {str(e)}"
        logger.error(f"Image gen Modal trigger failed for {job_id[:8]}: {error_msg}")
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/create", response_model=ImageGenResponse, status_code=202)
@limiter.limit("2/minute")
async def create_image_gen(
    body: ImageGenCreateRequest,
    request: Request,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Queue a Flux.1-schnell image generation job on Modal A10G.
    Returns immediately (202). Poll /api/image-gen/{job_id} for status.
    Completed jobs have image_url pointing to R2.
    """
    job_id = str(uuid.uuid4())

    # Credit check (5 credits per image) and deduct upfront
    credits_needed = 5 * body.num_images
    user_credits = await supabase.get_user_credits(user_id)
    if user_credits < credits_needed:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Image generation requires {credits_needed} credits ({body.num_images} images × 5 credits)."
        )

    deducted = await supabase.deduct_user_credits(user_id, credits_needed)
    if not deducted:
        raise HTTPException(
            status_code=402,
            detail="Credit deduction failed. Please try again."
        )

    # Persist job record
    await supabase.create_job({
        "job_id": job_id,
        "user_id": user_id,
        "status": "pending",
        "input_data": body.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "credits_estimated": credits_needed,
        "credits_deducted": True,
    })
    await supabase.update_job_status(job_id, "queued")

    # Dispatch to Modal background task with strong reference to prevent GC
    _dispatch_background_task(_trigger_modal_image_gen(job_id, user_id, body))

    return ImageGenResponse(
        job_id=job_id,
        status="queued",
        created_at=datetime.utcnow(),
        estimated_time_seconds=10,
    )


@router.get("/{job_id}", response_model=ImageGenResponse)
async def get_image_gen_status(
    job_id: str,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Poll job status. Frontend polls every 3 s until status = completed | failed.
    When completed, image_url contains the R2 CDN URL of the generated PNG.
    """
    job = await supabase.get_job(job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return ImageGenResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=datetime.fromisoformat(job["created_at"]),
        image_url=job.get("image_url"),
        error_message=job.get("error_message"),
        estimated_time_seconds=10,
    )
