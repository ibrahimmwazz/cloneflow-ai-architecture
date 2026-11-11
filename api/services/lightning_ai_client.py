"""
Lightning AI Client for GPU Inference
Routes TTS, Whisper, and Video generation to Lightning AI free GPU (80 hrs/month)
"""

import os
import httpx
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightningAIClient:
    """Client for Lightning AI GPU inference endpoints with connection pooling."""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("LIGHTNING_AI_API_KEY")
        self.base_url = base_url or os.getenv("LIGHTNING_AI_BASE_URL", "").rstrip("/")

        # Connection pool: 50 max, 20 keep-alive — handles thousands of users
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
        self._client = httpx.AsyncClient(timeout=120.0, limits=limits)

        if not self.api_key:
            logger.warning("LIGHTNING_AI_API_KEY not set — GPU features unavailable")
        if not self.base_url:
            logger.warning("LIGHTNING_AI_BASE_URL not set — GPU features unavailable")

        logger.info("Lightning AI client initialized (connection pooling enabled)")

    async def _post(self, endpoint: str, payload: dict, timeout: float = 120.0) -> Dict[str, Any]:
        """Generic POST helper with auth, retry, and error handling."""
        if not self.base_url or not self.api_key:
            return {"success": False, "error": "Lightning AI not configured"}

        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Retry logic for transient failures
        for attempt in range(3):
            try:
                logger.info(f"Lightning AI POST {endpoint} (attempt {attempt + 1})")
                response = await self._client.post(url, json=payload, headers=headers, timeout=timeout)
                logger.info(f"Lightning AI response: {response.status_code}")

                if response.status_code in (200, 201):
                    return {"success": True, **response.json()}

                # If 502/503/504, retry; else fail fast
                if response.status_code not in (502, 503, 504):
                    return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}

                logger.warning(f"Retryable error {response.status_code}, attempt {attempt + 1}/3")

            except Exception as e:
                logger.error(f"Lightning AI request failed (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return {"success": False, "error": str(e)}

        return {"success": False, "error": "Max retries exceeded"}

    # ---------- Whisper ----------
    async def transcribe_audio(
        self,
        audio_url: str,
        language: Optional[str] = None,
        model: str = "base"
    ) -> Dict[str, Any]:
        """Transcribe audio on Lightning AI GPU using Whisper"""
        payload = {"audio_url": audio_url, "model": model}
        if language:
            payload["language"] = language
        return await self._post("/whisper", payload)

    # ---------- LivePortrait Video ----------
    async def generate_video(
        self,
        image_url: str,
        driving_video_url: str,
        fps: int = 30
    ) -> Dict[str, Any]:
        """Generate face animation on Lightning AI GPU using LivePortrait"""
        return await self._post("/liveportrait", {
            "image_url": image_url,
            "driving_video_url": driving_video_url,
            "fps": fps
        })

    async def health_check(self) -> Dict[str, Any]:
        """Check if Lightning AI endpoint is reachable"""
        if not self.base_url:
            return {"status": "not_configured"}
        try:
            response = await self._client.get(f"{self.base_url}/health", timeout=10.0)
            return {"status": "healthy" if response.status_code == 200 else "unhealthy"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    async def close(self):
        """Close connection pool gracefully."""
        await self._client.aclose()
