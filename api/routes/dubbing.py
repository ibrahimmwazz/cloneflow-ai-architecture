"""
CloneFlow AI - Dubbing Router
Full dubbing pipeline: Whisper transcription → NLLB translation → Fish Speech TTS → Wav2Lip lip-sync
"""

import os
import uuid
import logging
import httpx
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from pydantic import BaseModel, Field
from api.services.supabase_client import get_supabase, SupabaseClient
from api.dependencies.auth import require_user
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/dubbing", tags=["Dubbing"])
logger = logging.getLogger(__name__)


# ── Request / Response models ───────────────────────────────────────────────

class DubbingCreateRequest(BaseModel):
    video_url: str = Field(..., description="URL of the video to dub")
    target_language: str = Field("en", description="Target language code (e.g., en, es, fr, de)")
    voice_id: Optional[str] = Field("alex", description="Voice preset for TTS: alex, sarah, anna, bob, luna (optional if using voice_ref_url)")
    voice_ref_url: Optional[str] = Field(None, description="URL of voice reference file for cloning")


class DubbingResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    estimated_time_seconds: int = 300


# ── Trigger helper ───────────────────────────────────────────────────────────

async def _trigger_modal_dubbing(job_id: str, user_id: str,
                                body: DubbingCreateRequest,
                                supabase: SupabaseClient):
    """Fire-and-forget: call Modal trigger_dubbing endpoint."""
    modal_url = os.getenv("MODAL_DUBBING_URL", "")
    if not modal_url:
        error_msg = "MODAL_DUBBING_URL not set — dubbing service unavailable"
        logger.error(error_msg)
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(modal_url, json={
                "job_id": job_id,
                "user_id": user_id,
                "video_url": body.video_url,
                "target_language": body.target_language,
                "voice_id": body.voice_id,
                "voice_ref_url": body.voice_ref_url,
            })
        if r.status_code == 200:
            logger.info(f"Dubbing job {job_id[:8]} dispatched → Modal HTTP {r.status_code}")
        else:
            error_msg = f"Modal returned HTTP {r.status_code}: {r.text[:200]}"
            logger.error(f"Dubbing Modal trigger failed for {job_id[:8]}: {error_msg}")
            await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
    except httpx.TimeoutException:
        error_msg = "Modal request timed out (15s) - service may be cold-starting"
        logger.error(f"Dubbing Modal timeout for {job_id[:8]}: {error_msg}")
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})
    except Exception as e:
        error_msg = f"Modal trigger failed: {str(e)}"
        logger.error(f"Dubbing Modal trigger failed for {job_id[:8]}: {error_msg}")
        await supabase.update_job_status(job_id, "failed", {"error_message": error_msg})


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/create", response_model=DubbingResponse, status_code=202)
@limiter.limit("2/minute")
async def create_dubbing(
    body: DubbingCreateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Queue a full dubbing job on Modal T4.
    Pipeline: Whisper → NLLB-200 → Fish Speech → Wav2Lip
    Returns immediately (202). Poll /api/dubbing/{job_id} for status.
    """
    job_id = uuid.uuid4().hex

    # Credit check (20 credits per dubbing) and deduct upfront
    user_credits = await supabase.get_user_credits(user_id)
    if user_credits < 20:
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits. Dubbing requires 20 credits."
        )

    deducted = await supabase.deduct_user_credits(user_id, 20)
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
        "credits_estimated": 20,
        "credits_deducted": True,
    })
    await supabase.update_job_status(job_id, "queued")

    # Dispatch to Modal (async, does not block response)
    background_tasks.add_task(_trigger_modal_dubbing, job_id, user_id, body, supabase)

    return DubbingResponse(
        job_id=job_id,
        status="queued",
        created_at=datetime.utcnow(),
        estimated_time_seconds=300,
    )


@router.get("/{job_id}", response_model=DubbingResponse)
async def get_dubbing_status(
    job_id: str,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Poll job status. Frontend polls every 5 s until status = completed | failed.
    When completed, video_url contains the R2 CDN URL of the dubbed video.
    """
    job = await supabase.get_job(job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return DubbingResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=datetime.fromisoformat(job["created_at"]),
        video_url=job.get("video_url"),
        error_message=job.get("error_message"),
        estimated_time_seconds=300,
    )
