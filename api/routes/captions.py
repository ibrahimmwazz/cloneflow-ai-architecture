"""
Captions API Routes
Endpoints for automatic caption generation using Whisper
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/captions", tags=["captions"])


# Pydantic models
class CaptionRequest(BaseModel):
    audio_url: str
    language: Optional[str] = None
    output_format: str = "srt"


class CaptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    captions: str
    format: str


class CaptionSegment(BaseModel):
    start: float
    end: float
    text: str


@router.post("/generate")
async def generate_captions(request: CaptionRequest):
    """
    Generate captions from audio URL
    
    Args:
        request: Caption generation parameters
        
    Returns:
        Generated captions
    """
    try:
        # Validate audio URL
        if not request.audio_url:
            raise HTTPException(status_code=400, detail="Audio URL is required")
        
        # Validate output format
        valid_formats = ["srt", "vtt", "json", "text"]
        if request.output_format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Must be one of: {', '.join(valid_formats)}"
            )
        
        logger.info(f"Generating captions for audio: {request.audio_url}")
        
        # Call Whisper service on GPU
        from api.services.whisper_service import WhisperService
        whisper = WhisperService()
        
        result = await whisper.transcribe(
            audio_url=request.audio_url,
            language=request.language,
            task="transcribe",
            return_timestamps=True
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Transcription failed"))
        
        # Convert to requested format
        text = result.get("text", "")
        language = result.get("language", "en")
        duration = result.get("duration", 0.0)
        
        # Generate captions in requested format
        captions = text
        if request.output_format == "srt":
            captions = _text_to_srt(text, duration)
        elif request.output_format == "vtt":
            captions = _text_to_vtt(text, duration)
        
        return CaptionResponse(
            text=text,
            language=language,
            duration=duration,
            captions=captions,
            format=request.output_format
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate captions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/file")
async def generate_captions_from_file(
    audio_file: UploadFile = File(...),
    language: Optional[str] = None,
    output_format: str = "srt"
):
    """
    Generate captions from uploaded audio file
    
    Args:
        audio_file: Audio file to transcribe
        language: Language code (auto-detect if None)
        output_format: Output format (srt, vtt, json, text)
        
    Returns:
        Generated captions
    """
    try:
        # Validate file type
        allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg"]
        if audio_file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Must be one of: {', '.join(allowed_types)}"
            )
        
        logger.info(f"Generating captions from file: {audio_file.filename}")
        
        # In production, this would:
        # 1. Save uploaded file
        # 2. Call Whisper service
        # 3. Return captions
        
        placeholder_captions = """1
00:00:00,000 --> 00:00:05,000
This is a sample caption from uploaded file.
"""
        
        return CaptionResponse(
            text="Sample transcription from uploaded file.",
            language=language or "en",
            duration=5.0,
            captions=placeholder_captions,
            format=output_format
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate captions from file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/languages")
async def get_supported_languages():
    """
    Get list of supported languages for captioning
    
    Returns:
        List of supported languages
    """
    languages = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "ru", "name": "Russian"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ko", "name": "Korean"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ar", "name": "Arabic"},
        {"code": "hi", "name": "Hindi"},
    ]
    
    return languages


@router.get("/formats")
async def get_supported_formats():
    """
    Get list of supported caption formats
    
    Returns:
        List of supported formats
    """
    return [
        {"code": "srt", "name": "SubRip", "description": "Most common subtitle format"},
        {"code": "vtt", "name": "WebVTT", "description": "Web standard for video subtitles"},
        {"code": "json", "name": "JSON", "description": "Machine-readable format with timing"},
        {"code": "text", "name": "Plain Text", "description": "Simple text without timing"},
    ]


def _text_to_srt(text: str, duration: float) -> str:
    """Convert plain text to SRT format"""
    lines = text.split(". ")
    total_duration = duration
    line_duration = total_duration / max(len(lines), 1)
    
    srt_output = []
    for i, line in enumerate(lines):
        start_time = i * line_duration
        end_time = (i + 1) * line_duration
        
        start_formatted = _format_srt_time(start_time)
        end_formatted = _format_srt_time(end_time)
        
        srt_output.append(f"{i + 1}")
        srt_output.append(f"{start_formatted} --> {end_formatted}")
        srt_output.append(line.strip() + ("." if not line.endswith(".") else ""))
        srt_output.append("")
    
    return "\n".join(srt_output)


def _text_to_vtt(text: str, duration: float) -> str:
    """Convert plain text to VTT format"""
    lines = text.split(". ")
    total_duration = duration
    line_duration = total_duration / max(len(lines), 1)
    
    vtt_output = ["WEBVTT", ""]
    
    for i, line in enumerate(lines):
        start_time = i * line_duration
        end_time = (i + 1) * line_duration
        
        start_formatted = _format_vtt_time(start_time)
        end_formatted = _format_vtt_time(end_time)
        
        vtt_output.append(f"{start_formatted} --> {end_formatted}")
        vtt_output.append(line.strip() + ("." if not line.endswith(".") else ""))
        vtt_output.append("")
    
    return "\n".join(vtt_output)


def _format_srt_time(seconds: float) -> str:
    """Format time for SRT (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format time for VTT (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
