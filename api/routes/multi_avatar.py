"""
Multi-Avatar API Routes
Endpoints for creating multi-avatar conversation videos.
Supports 3 styles: Elai.io, Colossyan, HeyGen.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum
import uuid
import logging

from api.services.multi_avatar_service import get_multi_avatar_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multi-avatar", tags=["multi-avatar"])


class ConversationStyle(str, Enum):
    ELAI = "elai"
    COLOSSYAN = "colossyan"
    HEYGEN = "heygen"


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SERIOUS = "serious"
    EXCITED = "excited"
    SAD = "sad"
    ANGRY = "angry"
    THOUGHTFUL = "thoughtful"
    SURPRISED = "surprised"


class AvatarConfig(BaseModel):
    """Configuration for an avatar in the conversation."""
    id: str = Field(..., description="Unique avatar identifier")
    name: str = Field(..., description="Display name")
    image_url: str = Field(..., description="URL of the avatar image")
    voice_id: Optional[str] = Field(None, description="TTS voice ID")
    driving_video_url: Optional[str] = Field(None, description="Driving video for animation")
    user_id: Optional[str] = Field(None, description="User ID for custom avatars")


class DialogueTurn(BaseModel):
    """A single turn in the conversation dialogue."""
    avatar_id: str = Field(..., description="Which avatar is speaking")
    text: str = Field(..., min_length=1, max_length=2000, description="What the avatar says")
    emotion: Emotion = Field(Emotion.NEUTRAL, description="Emotional tone")
    pause_after_ms: int = Field(500, ge=0, le=5000, description="Pause after this turn (ms)")

    @validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class SceneSettings(BaseModel):
    """Scene configuration for the conversation."""
    background_url: Optional[str] = Field(None, description="Background image/video URL")
    background_music_url: Optional[str] = Field(None, description="Background music URL")
    resolution: str = Field("1920x1080", description="Output resolution (WxH)")
    show_names: bool = Field(True, description="Show avatar names in video")


class ConversationRequest(BaseModel):
    """Request to create a multi-avatar conversation video."""
    avatars: List[AvatarConfig] = Field(..., min_items=2, max_items=4, description="Avatars in the scene")
    dialogue: List[DialogueTurn] = Field(..., min_items=1, description="Conversation dialogue")
    style: ConversationStyle = Field(ConversationStyle.ELAI, description="Layout style")
    scene: Optional[SceneSettings] = Field(None, description="Scene configuration")
    user_id: Optional[str] = Field(None, description="User ID for credit tracking")

    @validator("avatars")
    def unique_avatar_ids(cls, v):
        ids = [a.id for a in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Avatar IDs must be unique")
        return v

    @validator("dialogue")
    def valid_avatar_refs(cls, v, values):
        if "avatars" in values:
            valid_ids = {a.id for a in values["avatars"]}
            for turn in v:
                if turn.avatar_id not in valid_ids:
                    raise ValueError(f"Dialogue references unknown avatar: {turn.avatar_id}")
        return v


class ConversationResponse(BaseModel):
    """Response from multi-avatar conversation creation."""
    success: bool
    video_url: Optional[str] = None
    job_id: Optional[str] = None
    style: Optional[str] = None
    num_avatars: Optional[int] = None
    num_turns: Optional[int] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: int
    video_url: Optional[str] = None
    error: Optional[str] = None


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a multi-avatar conversation video.

    Supports three layout styles:
    - **elai**: Side-by-side conversational (2 avatars, like Elai.io)
    - **colossyan**: Multi-avatar scenario (up to 4, like Colossyan)
    - **heygen**: Avatar interaction with dynamic focus (like HeyGen)

    The pipeline:
    1. Generates TTS audio for each dialogue turn
    2. Creates individual avatar videos
    3. Removes backgrounds
    4. Mixes audio with timing
    5. Composites the final scene
    6. Uploads to cloud storage
    """
    try:
        job_id = f"multi_{uuid.uuid4().hex[:12]}"

        avatars_dict = [a.dict() for a in request.avatars]
        dialogue_dict = [d.dict() for d in request.dialogue]
        scene_dict = request.scene.dict() if request.scene else {}

        multi_avatar_service = get_multi_avatar_service()
        if not multi_avatar_service:
            raise HTTPException(status_code=503, detail="Multi-avatar service not available")

        background_tasks.add_task(
            multi_avatar_service.create_conversation,
            avatars=avatars_dict,
            dialogue=dialogue_dict,
            style=request.style.value,
            background_url=scene_dict.get("background_url"),
            background_music_url=scene_dict.get("background_music_url"),
            resolution=scene_dict.get("resolution", "1920x1080"),
            user_id=request.user_id,
            job_id=job_id
        )

        return ConversationResponse(
            success=True,
            job_id=job_id,
            style=request.style.value,
            num_avatars=len(request.avatars),
            num_turns=len(request.dialogue)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{job_id}", response_model=JobStatusResponse)
async def get_conversation_status(job_id: str):
    """
    Get the status of a multi-avatar conversation job.

    Returns progress percentage and final video URL when complete.
    """
    try:
        from api.services.redis_queue import get_redis_queue

        queue = await get_redis_queue()
        job = await queue.get_job_status(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobStatusResponse(
            job_id=job_id,
            status=job.get("status", "unknown"),
            progress=job.get("progress", 0),
            video_url=job.get("video_url"),
            error=job.get("error")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/styles")
async def get_available_styles():
    """
    Get available multi-avatar conversation styles with descriptions.
    """
    return {
        "styles": [
            {
                "id": "elai",
                "name": "Elai.io Style",
                "description": "Side-by-side conversational layout. Best for 2-avatar dialogues, interviews, and debates.",
                "max_avatars": 3,
                "best_for": ["interviews", "debates", "customer-support", "training-dialogues"],
                "layout": "horizontal side-by-side"
            },
            {
                "id": "colossyan",
                "name": "Colossyan Style",
                "description": "Scenario-based multi-avatar layout. Supports up to 4 avatars with flexible positioning.",
                "max_avatars": 4,
                "best_for": ["training-scenarios", "role-playing", "team-presentations", "educational-content"],
                "layout": "grid with flexible positioning"
            },
            {
                "id": "heygen",
                "name": "HeyGen Style",
                "description": "Interactive avatar layout with dynamic focus on the active speaker. Best for engaging content.",
                "max_avatars": 4,
                "best_for": ["marketing-videos", "product-demos", "social-media", "brand-content"],
                "layout": "dynamic with speaker focus"
            }
        ]
    }


@router.get("/emotions")
async def get_available_emotions():
    """Get available emotions for dialogue turns."""
    return {
        "emotions": [
            {"id": "neutral", "description": "Default neutral expression"},
            {"id": "happy", "description": "Cheerful and positive"},
            {"id": "serious", "description": "Professional and formal"},
            {"id": "excited", "description": "Energetic and enthusiastic"},
            {"id": "sad", "description": "Somber and empathetic"},
            {"id": "angry", "description": "Firm and assertive"},
            {"id": "thoughtful", "description": "Contemplative and reflective"},
            {"id": "surprised", "description": "Shocked or amazed"}
        ]
    }
