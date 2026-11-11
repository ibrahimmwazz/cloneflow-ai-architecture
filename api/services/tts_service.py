"""
TTS Service - SiliconFlow Fish Speech for all tiers
"""

import os
import logging
import uuid
from typing import Optional, Dict, Any
from api.services.cache_service import cache_service
from api.services.audio_processor import get_audio_processor
from api.utils.text_cleaner import clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSService:
    """TTS Service — SiliconFlow Fish Speech for free and premium"""

    def __init__(self):
        self.client = None  # SiliconFlowClient module was removed
        self.output_dir = "outputs/tts"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("TTSService initialized (SiliconFlow Fish Speech)")

    async def synthesize(
        self,
        text: str,
        language: str = "en",
        voice: str = "fishaudio:fish-speech-v1.5",
        plan: str = "free",
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """Synthesize speech using SiliconFlow Fish Speech."""
        try:
            # Normalize text before processing
            text = clean_text(text)

            # Check cache first
            cached = await cache_service.get_tts(text, voice, language)
            if cached:
                logger.info("TTS cache hit")
                return cached

            # Process long text through audio processor
            if len(text) > 500:
                audio_processor = get_audio_processor()
                if not audio_processor:
                    return {"success": False, "error": "Audio processor not available"}
                result = await audio_processor.process_long_text(
                    text=text, voice=voice, language=language, speed=speed
                )
            else:
                result = await self.client.generate_tts(
                    text=text, voice=voice, speed=speed
                )

            if not result.get("success"):
                return result

            # Save audio bytes to file
            if "audio_content" in result:
                filename = f"tts_{uuid.uuid4().hex[:10]}.mp3"
                filepath = os.path.join(self.output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(result["audio_content"])

                result["audio_url"] = f"/outputs/tts/{filename}"
                result["local_path"] = filepath
                del result["audio_content"]

            output = {
                "success": True,
                "audio_url": result.get("audio_url"),
                "local_path": result.get("local_path"),
                "text": text,
                "language": language,
                "voice": voice,
                "plan": plan,
                "provider": result.get("provider", "siliconflow_fishspeech")
            }

            # Cache result
            await cache_service.set_tts(text, voice, language, output)

            return output

        except Exception as e:
            logger.error(f"TTS service error: {e}")
            return {"success": False, "error": str(e), "text": text}

    async def get_voices(self, plan: str = "free") -> list:
        return [
            {"id": "fishaudio:fish-speech-v1.5", "name": "Fish Speech v1.5", "quality": "High"},
            {"id": "cosyvoice2", "name": "CosyVoice 2", "quality": "Premium"}
        ]


# Lazy instantiation - only create when needed to avoid startup failures
tts_service = None

def get_tts_service():
    """Get TTS service instance (lazy initialization)."""
    global tts_service
    if tts_service is None:
        try:
            tts_service = TTSService()
        except Exception as e:
            logger.warning(f"Failed to initialize TTSService: {e}")
            return None
    return tts_service
