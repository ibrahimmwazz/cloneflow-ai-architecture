"""
Avatar API Routes
Endpoints for stock avatar library and custom avatar management
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks, Request
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import uuid
import logging
import httpx
from api.dependencies.auth import require_user, optional_user
from api.services.supabase_client import get_supabase, SupabaseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/avatars", tags=["avatars"])


# Pydantic models
class Avatar(BaseModel):
    id: str
    name: str
    description: str
    image_url: str
    category: Optional[str] = "professional"
    gender: Optional[str] = "unspecified"
    ethnicity: Optional[str] = "unspecified"
    age_group: Optional[str] = "adult"
    is_stock: bool = True
    created_at: str


class AvatarGenerationRequest(BaseModel):
    prompt: str
    style: Optional[str] = "realistic"
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    seed: Optional[int] = None


class CustomAvatarTrainingRequest(BaseModel):
    name: str
    description: Optional[str] = None
    num_epochs: Optional[int] = 10
    learning_rate: Optional[float] = 1e-4


# In-memory storage for development (replace with database in production)
STOCK_AVATARS = [
    {
        "id": "stock_001",
        "name": "Professional Woman - Brown Hair",
        "description": "Young woman with short brown hair in business casual attire",
        "image_url": "https://via.placeholder.com/1024x1024?text=Avatar+001",
        "category": "professional",
        "gender": "female",
        "ethnicity": "caucasian",
        "age_group": "young_adult",
        "is_stock": True,
        "created_at": "2026-04-23"
    },
    {
        "id": "stock_002",
        "name": "Professional Man - Black Hair",
        "description": "Young man with black hair and glasses wearing a suit",
        "image_url": "https://via.placeholder.com/1024x1024?text=Avatar+002",
        "category": "professional",
        "gender": "male",
        "ethnicity": "asian",
        "age_group": "young_adult",
        "is_stock": True,
        "created_at": "2026-04-23"
    },
    {
        "id": "stock_003",
        "name": "Professional Woman - Blonde Hair",
        "description": "Middle-aged woman with blonde hair in business attire",
        "image_url": "https://via.placeholder.com/1024x1024?text=Avatar+003",
        "category": "professional",
        "gender": "female",
        "ethnicity": "caucasian",
        "age_group": "middle_aged",
        "is_stock": True,
        "created_at": "2026-04-23"
    },
]


@router.get("/stock", response_model=List[Avatar])
async def get_stock_avatars(
    category: Optional[str] = None,
    gender: Optional[str] = None,
    ethnicity: Optional[str] = None,
    limit: int = 50
):
    """
    Get stock avatars with optional filters
    
    Args:
        category: Filter by category (professional, casual, etc.)
        gender: Filter by gender
        ethnicity: Filter by ethnicity
        limit: Maximum number of results
        
    Returns:
        List of stock avatars
    """
    avatars = STOCK_AVATARS.copy()
    
    # Apply filters
    if category:
        avatars = [a for a in avatars if a.get("category") == category]
    if gender:
        avatars = [a for a in avatars if a.get("gender") == gender]
    if ethnicity:
        avatars = [a for a in avatars if a.get("ethnicity") == ethnicity]
    
    # Limit results
    avatars = avatars[:limit]
    
    return avatars


@router.get("/stock/{avatar_id}", response_model=Avatar)
async def get_stock_avatar(avatar_id: str):
    """
    Get a specific stock avatar by ID
    
    Args:
        avatar_id: Avatar ID
        
    Returns:
        Avatar details
    """
    avatar = next((a for a in STOCK_AVATARS if a["id"] == avatar_id), None)
    
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    
    return avatar


@router.post("/generate")
async def generate_avatar(
    request: AvatarGenerationRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Generate a custom avatar image using Flux.1-schnell on Modal T4 GPU.
    Returns job_id — poll /api/jobs/{job_id} until completed, then read video_url for the image URL.
    """
    job_id = uuid.uuid4().hex
    modal_url = os.getenv("MODAL_IMGGEN_URL", "")

    if not modal_url:
        raise HTTPException(status_code=503, detail="Image generation service not configured. MODAL_IMGGEN_URL not set.")

    # Create DB record so Modal can notify back (/api/jobs/{job_id}/update)
    await supabase.create_job({
        "job_id":            job_id,
        "user_id":           user_id,
        "status":            "pending",
        "input_data":        {"type": "image_gen", "prompt": request.prompt},
        "created_at":        datetime.utcnow().isoformat(),
        "credits_estimated": 5,
    })
    await supabase.update_job_status(job_id, "queued")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(modal_url, json={
                "job_id":  job_id,
                "user_id": user_id,
                "prompt":  request.prompt,
                "style":   request.style or "realistic",
                "width":   request.width or 768,
                "height":  request.height or 768,
                "seed":    request.seed or 42,
            })
        if r.status_code == 200:
            logger.info(f"Avatar image gen queued on Modal Flux: {job_id}")
            return {
                "id":          f"custom_{job_id[:8]}",
                "name":        "AI Generated Avatar",
                "description": request.prompt,
                "image_url":   None,
                "job_id":      job_id,
                "status":      "queued",
                "engine":      "flux-schnell",
                "category":    "custom",
                "is_stock":    False,
                "created_at":  datetime.utcnow().isoformat(),
            }
        else:
            raise HTTPException(status_code=502, detail=f"Modal service returned HTTP {r.status_code}")
    except Exception as e:
        logger.error(f"Modal image gen trigger failed: {e}")
        raise HTTPException(status_code=502, detail=f"Image generation service unavailable: {str(e)}")


@router.post("/custom/train")
async def train_custom_avatar(
    request: CustomAvatarTrainingRequest,
    images: List[UploadFile] = File(...)
):
    """
    Train a custom avatar from user images
    
    Args:
        request: Training parameters
        images: User images for training
        
    Returns:
        Training job details
    """
    try:
        # Validate images
        if len(images) < 5:
            raise HTTPException(
                status_code=400,
                detail="At least 5 images required for training"
            )
        
        if len(images) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 images allowed for training"
            )
        
        logger.info(f"Starting training for custom avatar: {request.name}")
        logger.info(f"Number of training images: {len(images)}")
        
        # In production, this would:
        # 1. Save images to storage
        # 2. Prepare training data
        # 3. Train LoRA
        # 4. Return training job ID
        
        # Placeholder response
        training_job = {
            "job_id": f"train_{os.urandom(8).hex()}",
            "name": request.name,
            "description": request.description,
            "status": "pending",
            "num_images": len(images),
            "num_epochs": request.num_epochs,
            "learning_rate": request.learning_rate,
            "estimated_time": "30-60 minutes",
            "created_at": "2026-04-23"
        }
        
        return training_job
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/custom/{job_id}")
async def get_training_status(job_id: str):
    """
    Get training job status
    
    Args:
        job_id: Training job ID
        
    Returns:
        Training job status
    """
    # Placeholder - in production, query database for job status
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "avatar_id": f"custom_{job_id}",
        "created_at": "2026-04-23",
        "completed_at": "2026-04-23"
    }


@router.get("/custom")
async def list_custom_avatars(user_id: Optional[str] = None):
    """
    List custom avatars
    
    Args:
        user_id: Filter by user ID (if provided)
        
    Returns:
        List of custom avatars
    """
    # Placeholder - in production, query database
    return []


@router.delete("/custom/{avatar_id}")
async def delete_custom_avatar(avatar_id: str):
    """
    Delete a custom avatar
    
    Args:
        avatar_id: Avatar ID to delete
        
    Returns:
        Deletion confirmation
    """
    # Placeholder - in production, delete from database and storage
    return {
        "message": f"Avatar {avatar_id} deleted successfully"
    }


@router.get("/categories")
async def get_avatar_categories():
    """
    Get available avatar categories
    
    Returns:
        List of categories
    """
    return [
        {"id": "professional", "name": "Professional", "count": 30},
        {"id": "casual", "name": "Casual", "count": 10},
        {"id": "creative", "name": "Creative", "count": 5},
        {"id": "custom", "name": "Custom", "count": 0}
    ]


@router.get("/filters")
async def get_avatar_filters():
    """
    Get available filter options
    
    Returns:
        Available filter options
    """
    return {
        "genders": ["male", "female", "unspecified"],
        "ethnicities": ["caucasian", "asian", "black", "hispanic", "middle_eastern", "unspecified"],
        "age_groups": ["young_adult", "middle_aged", "senior", "unspecified"],
        "categories": ["professional", "casual", "creative", "custom"]
    }
