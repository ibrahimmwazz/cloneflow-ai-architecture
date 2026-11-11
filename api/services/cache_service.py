"""
CloneFlow AI - Cache Service
Redis-based caching with connection pooling for thousands of users
"""

import json
import hashlib
import logging
from typing import Optional, Any, Dict
from api.services.redis_queue import get_redis_queue

logger = logging.getLogger(__name__)


class CacheService:
    """High-performance caching layer for API responses and generated assets."""

    # TTL constants (seconds)
    TTL_IMAGE = 3600       # 1 hour
    TTL_TTS = 86400        # 24 hours
    TTL_WHISPER = 86400    # 24 hours
    TTL_JOB = 86400 * 7    # 7 days
    TTL_CREDIT = 60        # 1 minute (short, for atomic deductions)

    def __init__(self):
        self.redis = get_redis_queue()

    def _make_key(self, prefix: str, *parts) -> str:
        """Create a deterministic cache key."""
        raw = ":".join(str(p) for p in parts)
        # Hash long keys to stay within Redis limits
        if len(raw) > 200:
            raw = hashlib.sha256(raw.encode()).hexdigest()
        return f"{prefix}:{raw}"

    async def get_image(self, prompt: str, model: str, width: int, height: int) -> Optional[Dict]:
        """Fetch cached image generation result."""
        key = self._make_key("img", model, width, height, prompt)
        return await self.redis.cache_get(key)

    async def set_image(self, prompt: str, model: str, width: int, height: int, data: Dict) -> None:
        """Cache image generation result."""
        key = self._make_key("img", model, width, height, prompt)
        await self.redis.cache_set(key, data, ttl=self.TTL_IMAGE)

    async def get_tts(self, text: str, voice: str, language: str) -> Optional[Dict]:
        """Fetch cached TTS result."""
        key = self._make_key("tts", language, voice, text)
        return await self.redis.cache_get(key)

    async def set_tts(self, text: str, voice: str, language: str, data: Dict) -> None:
        """Cache TTS result."""
        key = self._make_key("tts", language, voice, text)
        await self.redis.cache_set(key, data, ttl=self.TTL_TTS)

    async def get_whisper(self, audio_hash: str) -> Optional[Dict]:
        """Fetch cached Whisper transcription."""
        key = self._make_key("whisper", audio_hash)
        return await self.redis.cache_get(key)

    async def set_whisper(self, audio_hash: str, data: Dict) -> None:
        """Cache Whisper transcription."""
        key = self._make_key("whisper", audio_hash)
        await self.redis.cache_set(key, data, ttl=self.TTL_WHISPER)

    async def get_job(self, job_id: str) -> Optional[Dict]:
        """Fetch cached job status/result."""
        key = self._make_key("job", job_id)
        return await self.redis.cache_get(key)

    async def set_job(self, job_id: str, data: Dict, ttl: int = None) -> None:
        """Cache job status/result."""
        key = self._make_key("job", job_id)
        await self.redis.cache_set(key, data, ttl=ttl or self.TTL_JOB)

    async def delete_job(self, job_id: str) -> None:
        """Remove cached job."""
        key = self._make_key("job", job_id)
        await self.redis.cache_delete(key)

    # ---------- Credit System ----------

    async def get_credits(self, user_id: str) -> Dict:
        """Get user's remaining credits (with short TTL)."""
        key = self._make_key("credits", user_id)
        data = await self.redis.cache_get(key)
        if data:
            return data
        # Return empty if not cached — caller should fetch from DB
        return {"images": 0, "audio": 0, "video": 0, "reset_at": None}

    async def set_credits(self, user_id: str, credits: Dict) -> None:
        """Cache user credits."""
        key = self._make_key("credits", user_id)
        await self.redis.cache_set(key, credits, ttl=self.TTL_CREDIT)

    async def deduct_credit(self, user_id: str, action: str, cost: int = 1) -> bool:
        """
        Atomically deduct a credit using Redis INCR/DECR.
        Returns True if deduction succeeded, False if insufficient.
        """
        redis = await self.redis.client
        key = self._make_key("credits", user_id, action)

        # Try to decrement; if key doesn't exist, let caller handle from DB
        current = await redis.get(key)
        if current is None:
            return False  # Not cached — fetch from DB first

        current = int(current)
        if current < cost:
            return False

        await redis.decrby(key, cost)
        return True

    async def reset_daily_credits(self, user_id: str, limits: Dict[str, int]) -> None:
        """Reset user's daily credit counters."""
        redis = await self.redis.client
        key = self._make_key("credits", user_id)
        # 24-hour TTL for the whole bucket
        await redis.setex(key, 86400, json.dumps(limits))


# Global instance
cache_service = CacheService()
