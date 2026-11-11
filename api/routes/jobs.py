"""
CloneFlow AI - Jobs Router
Handles job creation, status checking, and management
"""

import hmac
import json
import logging
import os
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Header
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum
from datetime import datetime
import uuid
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

logger = logging.getLogger(__name__)
from api.services.supabase_client import get_supabase, SupabaseClient
from api.services.redis_queue import get_redis_queue, RedisQueue
from api.config import settings
from api.dependencies.auth import require_user, optional_user


def _verify_worker_secret(x_worker_secret: Optional[str] = Header(None)) -> None:
    """
    Dependency: ensures the caller is the Modal worker, not a random request.
    
    SECURITY: Strict validation - rejects if WORKER_SECRET is not set or doesn't match.
    Uses constant-time comparison to prevent timing side-channels.
    """
    expected = settings.WORKER_SECRET or os.getenv("WORKER_SECRET", "")
    if not expected:
        logger.error("WORKER_SECRET is not configured on backend")
        raise HTTPException(
            status_code=503,
            detail="Worker authentication not configured"
        )

    if not x_worker_secret:
        logger.warning("Callback request missing X-Worker-Secret header")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing worker secret header"
        )

    if not hmac.compare_digest(x_worker_secret, expected):
        logger.warning("Callback request rejected due to invalid worker secret")
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid worker secret"
        )

router = APIRouter()


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnimationModel(str, Enum):
    """2026 I2V Models with MoE architecture."""
    LTX2 = "ltx2"           # Preferred: Native 720p, efficient on 12-24GB VRAM
    WAN2_2 = "wan2.2"       # Wan2.2-Turbo: 18x faster than legacy models
    LIVE_PORTRAIT = "liveportrait"  # Legacy fallback
    SADTALKER = "sadtalker"         # Legacy low-VRAM fallback
    MUSETALK = "musetalk"           # High-quality lip-sync alternative


class AspectRatio(str, Enum):
    """Supported aspect ratios for different platforms."""
    TIKTOK = "9:16"      # 1080x1920
    YOUTUBE = "16:9"     # 1920x1080
    INSTAGRAM = "1:1"    # 1080x1080


class JobCreateRequest(BaseModel):
    """Request model for creating a new avatar generation job (2026 Standard)."""
    image_url: str = Field(..., description="URL of the source image for avatar")
    text: str = Field(..., min_length=1, max_length=5000, description="Text for the avatar to speak")
    voice_id: Optional[str] = Field("alex", description="Preset voice: alex, sarah, bob, luna, anna")
    voice_ref_url: Optional[str] = Field(None, description="R2 URL of voice sample for Fish Speech cloning")
    
    # 2026 Model Options
    model: AnimationModel = Field(
        AnimationModel.LTX2, 
        description="I2V animation model: LTX-2 (default), Wan2.2-Turbo, LivePortrait, or SadTalker"
    )
    enhance_face: bool = Field(True, description="Apply GFPGAN/CodeFormer face enhancement")
    output_resolution: str = Field("720p", description="Output video resolution: 480p, 720p (default), 1080p")
    aspect_ratio: AspectRatio = Field(
        AspectRatio.YOUTUBE, 
        description="Aspect ratio: 9:16 (TikTok), 16:9 (YouTube), 1:1 (Instagram)"
    )
    use_upscaling: bool = Field(
        False, 
        description="Use Real-ESRGAN upscaling to 4K (only with 1080p output)"
    )
    
    @validator('image_url')
    def validate_image_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('image_url must be a valid HTTP/HTTPS URL')
        return v


class JobResponse(BaseModel):
    """Response model for job operations."""
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    credits_used: int = 0
    estimated_time_seconds: int = 60


class JobListResponse(BaseModel):
    """Response model for listing jobs."""
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int


@router.post("/create", response_model=JobResponse, status_code=202)
@limiter.limit("2/minute")
async def create_job(
    body: JobCreateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase),
    redis: RedisQueue = Depends(get_redis_queue)
):
    """
    Create a new avatar generation job.
    
    The job is queued for processing and will be executed asynchronously
    by the Lightning AI GPU worker.
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Check user credits and deduct upfront
    user_credits = await supabase.get_user_credits(user_id)
    estimated_credits = 10  # Base credits for video generation
    
    if user_credits < estimated_credits:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Required: {estimated_credits}, Available: {user_credits}"
        )
        
    deducted = await supabase.deduct_user_credits(user_id, estimated_credits)
    if not deducted:
        raise HTTPException(
            status_code=402,
            detail="Credit deduction failed. Please try again."
        )
    
    # Create job record in database
    job_data = {
        "job_id": job_id,
        "user_id": user_id,
        "status": JobStatus.PENDING.value,
        "input_data": body.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "credits_estimated": estimated_credits,
        "credits_deducted": True
    }
    
    await supabase.create_job(job_data)
    
    # Queue job in Redis (2026 Standard parameters)
    job_payload = {
        "job_id":       job_id,
        "user_id":      user_id,
        "image_url":    body.image_url,
        "text":         body.text,
        "voice_id":     body.voice_id or "alex",
        "voice_ref_url": body.voice_ref_url,
        "model":        body.model.value,
        "enhance_face": body.enhance_face,
    }
    
    await redis.enqueue_job("avatar_generation", job_payload)

    # Trigger Modal worker directly (serverless, auto-wakes on demand)
    background_tasks.add_task(_trigger_modal_worker, job_payload)

    # Update status to queued
    await supabase.update_job_status(job_id, JobStatus.QUEUED.value)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.utcnow(),
        estimated_time_seconds=90
    )


async def _trigger_modal_worker(job_payload: dict):
    """
    Calls the Modal serverless worker directly.
    Modal auto-wakes a T4 GPU — no manual start needed.
    Falls back silently if Modal is not configured (Lightning AI worker picks it up instead).
    """
    modal_url = os.getenv("MODAL_WORKER_URL", "")
    if not modal_url:
        logger.info("MODAL_WORKER_URL not set — Lightning AI polling worker will handle job")
        return

    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                modal_url,
                json={
                    "job_id":        job_payload["job_id"],
                    "user_id":       job_payload["user_id"],
                    "image_url":     job_payload["image_url"],
                    "text":          job_payload["text"],
                    "voice_id":      job_payload.get("voice_id", "alex"),
                    "voice_ref_url": job_payload.get("voice_ref_url"),
                    "model":         job_payload.get("model", "sadtalker"),
                    "enhance_face":  job_payload.get("enhance_face", True),
                },
            )
        logger.info(f"Modal worker triggered for job {job_payload['job_id'][:8]}")
    except Exception as e:
        logger.warning(f"Modal trigger failed (Lightning AI fallback active): {e}")


@router.get("/queued/list")
async def list_queued_jobs(
    limit: int = 5,
    supabase: SupabaseClient = Depends(get_supabase),
    _: None = Depends(_verify_worker_secret),
):
    """Called by the Modal/Lightning AI worker to poll for pending jobs."""
    jobs_queued = await supabase.list_jobs_by_status("queued", limit=limit)
    jobs_pending = await supabase.list_jobs_by_status("pending", limit=limit)
    all_jobs = (jobs_queued or []) + (jobs_pending or [])
    return {"jobs": all_jobs, "count": len(all_jobs)}


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    user_id: Optional[str] = Depends(optional_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Get the current status of a job.
    Authenticated users see their own jobs. Unauthenticated requests
    can read completed jobs (for public share pages).
    """
    if user_id:
        job = await supabase.get_job(job_id, user_id)
    else:
        job = await supabase.get_job_public(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        created_at=datetime.fromisoformat(job["created_at"]),
        updated_at=datetime.fromisoformat(job["updated_at"]) if job.get("updated_at") else None,
        video_url=job.get("video_url"),
        error_message=job.get("error_message"),
        credits_used=job.get("credits_used", 0),
        estimated_time_seconds=60
    )


@router.get("", response_model=JobListResponse)
@router.get("/", response_model=JobListResponse)
async def list_jobs(
    page: int = 1,
    page_size: int = 20,
    status: Optional[JobStatus] = None,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    List all jobs for the authenticated user.
    
    Supports pagination and filtering by status.
    """
    jobs, total = await supabase.list_user_jobs(
        user_id=user_id,
        page=page,
        page_size=page_size,
        status_filter=status.value if status else None
    )
    
    return JobListResponse(
        jobs=[
            JobResponse(
                job_id=job["job_id"],
                status=JobStatus(job["status"]),
                created_at=datetime.fromisoformat(job["created_at"]),
                updated_at=datetime.fromisoformat(job["updated_at"]) if job.get("updated_at") else None,
                video_url=job.get("video_url"),
                error_message=job.get("error_message"),
                credits_used=job.get("credits_used", 0),
                estimated_time_seconds=60
            )
            for job in jobs
        ],
        total=total,
        page=page,
        page_size=page_size
    )


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Cancel a pending or processing job.
    
    Cannot cancel completed or already cancelled jobs.
    """
    job = await supabase.get_job(job_id, user_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    current_status = JobStatus(job["status"])
    
    if current_status in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {current_status.value}"
        )
    
    # Update status and credits_deducted
    await supabase.update_job(job_id, {
        "status": JobStatus.CANCELLED.value,
        "credits_deducted": False
    })
    
    # Refund credits if applicable
    if job.get("credits_deducted") is True:
        await supabase.add_user_credits(user_id, job.get("credits_estimated", 10))
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.CANCELLED,
        created_at=datetime.fromisoformat(job["created_at"]),
        updated_at=datetime.utcnow()
    )


class JobUpdateRequest(BaseModel):
    """Payload sent by the Lightning AI worker to update a job."""
    status: JobStatus
    video_url: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[int] = None


@router.post("/{job_id}/update")
async def update_job_status(
    job_id: str,
    payload: JobUpdateRequest,
    supabase: SupabaseClient = Depends(get_supabase),
    redis: RedisQueue = Depends(get_redis_queue),
    _: None = Depends(_verify_worker_secret),
):
    """
    Called by the Modal GPU worker to update job status.
    Protected by WORKER_SECRET header — rejects requests from the public internet.
    
    Also publishes status updates to Redis Pub/Sub for real-time SSE streaming.
    """
    update_data: dict = {
        "status": payload.status.value,
        "updated_at": datetime.utcnow().isoformat(),
    }
    if payload.video_url:
        update_data["video_url"] = payload.video_url
    if payload.error:
        update_data["error_message"] = payload.error

    # Handle credit updates based on job status
    job = await supabase.get_job_public(job_id)
    if job:
        user_id = job.get("user_id")
        credits = job.get("credits_estimated", 0)
        
        if payload.status.value == "completed":
            # Already deducted upfront, so just record credits_used matching estimated
            update_data["credits_used"] = credits
        elif payload.status.value == "failed":
            # Refund upfront deduction
            if user_id and credits > 0 and job.get("credits_deducted") is True:
                await supabase.add_user_credits(user_id, credits)
                update_data["credits_deducted"] = False
                logger.info(f"Refunded {credits} credits for failed job {job_id}")

    # Update Supabase
    await supabase.update_job(job_id, update_data)
    logger.info(f"Worker updated job {job_id} -> {payload.status.value}")
    
    # Publish to Redis Pub/Sub for real-time SSE streaming
    try:
        client = await redis.client
        if client:
            channel_name = f"job_status:{job_id}"
            event_data = {
                "job_id": job_id,
                "status": payload.status.value,
                "video_url": payload.video_url,
                "error_message": payload.error,
                "progress": payload.progress,
                "timestamp": datetime.utcnow().isoformat()
            }
            await client.publish(channel_name, json.dumps(event_data))
            logger.info(f"Published job status update to Redis channel: {channel_name}")
        else:
            logger.warning("Redis client not available, skipping Pub/Sub publish")
    except Exception as e:
        logger.error(f"Failed to publish job status to Redis: {e}")
    
    return {"job_id": job_id, "status": payload.status.value}


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Delete a job record.
    
    Only completed, failed, or cancelled jobs can be deleted.
    """
    job = await supabase.get_job(job_id, user_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    current_status = JobStatus(job["status"])
    
    if current_status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active jobs. Cancel first."
        )
    
    await supabase.delete_job(job_id, user_id)
    
    return {"message": "Job deleted successfully", "job_id": job_id}
