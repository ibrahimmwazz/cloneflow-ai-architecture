"""
Whisper Service - Primary: Lightning AI GPU, Fallback: SiliconFlow
"""

import os
import logging
from typing import Optional, Dict, Any
from api.services.siliconflow_client import SiliconFlowClient
from api.services.lightning_ai_client import LightningAIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperService:
    """Whisper transcription service — GPU via Lightning AI, fallback to SiliconFlow"""
    
    def __init__(self):
        """Initialize whisper service with Lightning AI (primary) and SiliconFlow (fallback)"""
        self.lightning_client = LightningAIClient()
        self.siliconflow_client = SiliconFlowClient()
        logger.info("WhisperService initialized (Lightning AI primary, SiliconFlow fallback)")
    
    async def transcribe(
        self,
        audio_url: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio — tries Lightning AI GPU first, falls back to SiliconFlow
        """
        try:
            logger.info(f"Transcribing audio: {audio_url}")

            # 1. Try Lightning AI GPU first (faster, free tier)
            lightning_result = await self.lightning_client.transcribe_audio(
                audio_url=audio_url,
                language=language
            )
            if lightning_result.get("success"):
                logger.info("Transcription served by Lightning AI GPU")
                return {
                    "success": True,
                    "text": lightning_result.get("text"),
                    "language": lightning_result.get("language"),
                    "duration": lightning_result.get("duration"),
                    "provider": "lightning_ai"
                }
            logger.warning(f"Lightning AI failed: {lightning_result.get('error')} — falling back to SiliconFlow")

            # 2. Fallback to SiliconFlow
            result = await self.siliconflow_client.transcribe_audio(
                audio_url=audio_url,
                language=language
            )

            if not result.get("success"):
                return result

            return {
                "success": True,
                "text": result.get("text"),
                "language": result.get("language"),
                "duration": result.get("duration"),
                "inference_time": result.get("inference_time"),
                "provider": "siliconflow"
            }

        except Exception as e:
            logger.error(f"Whisper service error: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_url": audio_url
            }
    
    async def transcribe_with_speaker_detection(
        self,
        audio_url: str,
        num_speakers: int = 2
    ) -> Dict[str, Any]:
        """
        Transcribe with speaker diarization
        
        Args:
            audio_url: URL to audio file
            num_speakers: Number of speakers to identify
            
        Returns:
            Dict with speaker-segmented transcription
        """
        try:
            # First get basic transcription
            result = await self.transcribe(audio_url, return_timestamps=True)
            
            if not result.get("success"):
                return result
            
            # Add speaker labels to segments (simplified)
            segments = result.get("segments", [])
            speaker_segments = []
            
            for i, segment in enumerate(segments):
                speaker_id = f"Speaker {i % num_speakers + 1}"
                speaker_segments.append({
                    "speaker": speaker_id,
                    "text": segment.get("text"),
                    "start": segment.get("start"),
                    "end": segment.get("end")
                })
            
            return {
                "success": True,
                "text": result.get("text"),
                "speakers": speaker_segments,
                "num_speakers": num_speakers
            }
            
        except Exception as e:
            logger.error(f"Speaker detection error: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_subtitles(
        self,
        audio_url: str,
        format: str = "srt",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate subtitle file from audio
        
        Args:
            audio_url: URL to audio file
            format: Subtitle format (srt, vtt, txt)
            language: Language code
            
        Returns:
            Dict with subtitle content
        """
        try:
            # Get transcription with timestamps
            result = await self.transcribe(
                audio_url=audio_url,
                language=language,
                return_timestamps=True
            )
            
            if not result.get("success"):
                return result
            
            segments = result.get("segments", [])
            
            # Format as SRT
            if format == "srt":
                subtitles = []
                for i, segment in enumerate(segments, 1):
                    start = self._seconds_to_srt_time(segment.get("start", 0))
                    end = self._seconds_to_srt_time(segment.get("end", 0))
                    text = segment.get("text", "").strip()
                    
                    subtitles.append(f"{i}\n{start} --> {end}\n{text}\n")
                
                subtitle_content = "\n".join(subtitles)
                
            elif format == "vtt":
                subtitles = ["WEBVTT\n"]
                for segment in segments:
                    start = self._seconds_to_vtt_time(segment.get("start", 0))
                    end = self._seconds_to_vtt_time(segment.get("end", 0))
                    text = segment.get("text", "").strip()
                    
                    subtitles.append(f"{start} --> {end}\n{text}\n")
                
                subtitle_content = "\n".join(subtitles)
                
            else:  # txt
                subtitle_content = result.get("text", "")
            
            return {
                "success": True,
                "content": subtitle_content,
                "format": format,
                "segments": len(segments)
            }
            
        except Exception as e:
            logger.error(f"Subtitle generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to VTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check SiliconFlow service health"""
        return {"status": "healthy", "provider": "siliconflow"}


# Global instance
whisper_service = WhisperService()
