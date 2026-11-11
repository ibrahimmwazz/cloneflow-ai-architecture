"""
Scripts API Routes
Endpoints for AI-powered script generation using Llama 3
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scripts", tags=["scripts"])


# Pydantic models
class ScriptGenerationRequest(BaseModel):
    topic: str
    tone: str = "professional"
    length: str = "medium"
    language: str = "en"


class ScriptImprovementRequest(BaseModel):
    original_script: str
    feedback: str
    tone: Optional[str] = None


class ScriptResponse(BaseModel):
    topic: str
    tone: str
    length: str
    language: str
    script: str
    word_count: int
    estimated_duration: float
    placeholder: bool = False


@router.post("/generate", response_model=ScriptResponse)
async def generate_script(request: ScriptGenerationRequest):
    """
    Generate a script for avatar video using AI
    
    Args:
        request: Script generation parameters
        
    Returns:
        Generated script with metadata
    """
    try:
        # Validate inputs
        valid_tones = ["professional", "casual", "enthusiastic"]
        if request.tone not in valid_tones:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tone. Must be one of: {', '.join(valid_tones)}"
            )
        
        valid_lengths = ["short", "medium", "long"]
        if request.length not in valid_lengths:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid length. Must be one of: {', '.join(valid_lengths)}"
            )
        
        if len(request.topic) < 5:
            raise HTTPException(status_code=400, detail="Topic too short (min 5 characters)")
        
        if len(request.topic) > 500:
            raise HTTPException(status_code=400, detail="Topic too long (max 500 characters)")
        
        logger.info(f"Generating script for topic: {request.topic}")
        
        # In production, this would call the Llama service
        # For now, return a placeholder
        
        placeholder_script = f"""
Welcome to our video about {request.topic}.

In this presentation, we'll explore the key aspects of {request.topic} and how it impacts our daily lives. Understanding this subject is essential for making informed decisions and staying ahead in today's fast-paced world.

Let's dive into the main points and discover why {request.topic} matters to you.

Thank you for watching.
""".strip()
        
        word_count = len(placeholder_script.split())
        
        return ScriptResponse(
            topic=request.topic,
            tone=request.tone,
            length=request.length,
            language=request.language,
            script=placeholder_script,
            word_count=word_count,
            estimated_duration=word_count / 150,  # ~150 words per minute
            placeholder=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/improve")
async def improve_script(request: ScriptImprovementRequest):
    """
    Improve an existing script based on feedback
    
    Args:
        request: Script improvement parameters
        
    Returns:
        Improved script
    """
    try:
        # Validate inputs
        if len(request.original_script) < 10:
            raise HTTPException(status_code=400, detail="Original script too short")
        
        if len(request.feedback) < 5:
            raise HTTPException(status_code=400, detail="Feedback too short")
        
        logger.info("Improving script based on feedback")
        
        # In production, this would call the Llama service
        # For now, return the original script with a note
        
        return {
            "original_script": request.original_script,
            "improved_script": request.original_script,
            "feedback": request.feedback,
            "tone": request.tone,
            "placeholder": True,
            "note": "Llama service not connected - returning original script"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to improve script: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tones")
async def get_available_tones():
    """
    Get list of available script tones
    
    Returns:
        List of available tones
    """
    return [
        {
            "id": "professional",
            "name": "Professional",
            "description": "Formal and business-appropriate language"
        },
        {
            "id": "casual",
            "name": "Casual",
            "description": "Friendly and conversational language"
        },
        {
            "id": "enthusiastic",
            "name": "Enthusiastic",
            "description": "Energetic and exciting language"
        }
    ]


@router.get("/lengths")
async def get_available_lengths():
    """
    Get list of available script lengths
    
    Returns:
        List of available lengths
    """
    return [
        {
            "id": "short",
            "name": "Short",
            "description": "100-150 words (~30-45 seconds)",
            "word_range": [100, 150]
        },
        {
            "id": "medium",
            "name": "Medium",
            "description": "200-300 words (~45-90 seconds)",
            "word_range": [200, 300]
        },
        {
            "id": "long",
            "name": "Long",
            "description": "400-500 words (~90-120 seconds)",
            "word_range": [400, 500]
        }
    ]


@router.get("/templates")
async def get_script_templates():
    """
    Get pre-defined script templates
    
    Returns:
        List of script templates
    """
    return [
        {
            "id": "product_launch",
            "name": "Product Launch",
            "topic": "Introducing a new product",
            "tone": "enthusiastic",
            "length": "medium"
        },
        {
            "id": "tutorial",
            "name": "Tutorial",
            "topic": "How-to guide",
            "tone": "professional",
            "length": "long"
        },
        {
            "id": "announcement",
            "name": "Announcement",
            "topic": "Company announcement",
            "tone": "professional",
            "length": "short"
        },
        {
            "id": "testimonial",
            "name": "Testimonial",
            "topic": "Customer testimonial",
            "tone": "casual",
            "length": "medium"
        }
    ]
