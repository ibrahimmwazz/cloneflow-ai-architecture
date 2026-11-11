"""
TTS API Routes
Endpoints for multi-language text-to-speech
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])


# Pydantic models
class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker_id: Optional[str] = "af_heart"
    plan: str = "free"  # free or premium
    speed: float = 1.0


class TTSResponse(BaseModel):
    audio_url: str
    language: str
    duration: Optional[float] = None
    provider: str
    plan: str


class Language(BaseModel):
    code: str
    name: str
    native_name: str


# Supported languages
SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English", "native_name": "English"},
    {"code": "es", "name": "Spanish", "native_name": "Español"},
    {"code": "fr", "name": "French", "native_name": "Français"},
    {"code": "de", "name": "German", "native_name": "Deutsch"},
    {"code": "pt", "name": "Portuguese", "native_name": "Português"},
    {"code": "it", "name": "Italian", "native_name": "Italiano"},
    {"code": "pl", "name": "Polish", "native_name": "Polski"},
    {"code": "tr", "name": "Turkish", "native_name": "Türkçe"},
    {"code": "ru", "name": "Russian", "native_name": "Русский"},
    {"code": "nl", "name": "Dutch", "native_name": "Nederlands"},
    {"code": "ja", "name": "Japanese", "native_name": "日本語"},
    {"code": "ko", "name": "Korean", "native_name": "한국어"},
    {"code": "zh-cn", "name": "Chinese (Simplified)", "native_name": "中文（简体）"},
    {"code": "ar", "name": "Arabic", "native_name": "العربية"},
]


@router.get("/languages", response_model=list[Language])
async def get_supported_languages():
    """
    Get list of supported languages
    
    Returns:
        List of supported languages
    """
    return SUPPORTED_LANGUAGES


@router.get("/languages/{language_code}", response_model=Language)
async def get_language(language_code: str):
    """
    Get language details by code
    
    Args:
        language_code: Language code
        
    Returns:
        Language details
    """
    language = next((lang for lang in SUPPORTED_LANGUAGES if lang["code"] == language_code), None)
    
    if not language:
        raise HTTPException(status_code=404, detail="Language not found")
    
    return language


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text
    """
    try:
        from api.services.tts_service import get_tts_service
        
        tts_service = get_tts_service()
        if not tts_service:
            raise HTTPException(status_code=503, detail="TTS service not available")
        
        # Validate text length
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        if len(request.text) < 1:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Synthesizing speech in {request.language} via {request.plan} plan")
        
        # Call TTS service
        result = await tts_service.synthesize(
            text=request.text,
            language=request.language,
            voice=request.speaker_id,
            plan=request.plan,
            speed=request.speed
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Synthesis failed"))
        
        return TTSResponse(
            audio_url=result.get("audio_url"),
            language=result.get("language"),
            duration=result.get("duration"),
            provider=result.get("provider"),
            plan=result.get("plan")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to synthesize speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize/batch")
async def synthesize_batch(texts: list[str], language: str = "en"):
    """
    Synthesize multiple texts
    
    Args:
        texts: List of texts to synthesize
        language: Language code
        
    Returns:
        List of generated audio details
    """
    try:
        if len(texts) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 texts per batch")
        
        results = []
        for i, text in enumerate(texts):
            try:
                audio_url = f"https://via.placeholder.com/audio?text={language}_{i}"
                results.append({
                    "index": i,
                    "text": text,
                    "audio_url": audio_url,
                    "language": language,
                    "duration": 5.0
                })
            except Exception as e:
                logger.error(f"Failed to synthesize text {i}: {e}")
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to synthesize batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speakers")
async def get_available_speakers(plan: str = "free"):
    """
    Get list of available speaker voices
    """
    from api.services.tts_service import get_tts_service
    tts_service = get_tts_service()
    if not tts_service:
        return []
    return await tts_service.get_voices(plan=plan)


@router.get("/voices/custom")
async def get_custom_voices(user_id: Optional[str] = None):
    """
    Get custom voices for a user
    
    Args:
        user_id: Filter by user ID
        
    Returns:
        List of custom voices
    """
    # Placeholder - in production, query database for custom voices
    return []
