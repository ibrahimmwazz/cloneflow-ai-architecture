"""
CloneFlow AI - Redis Queue Service
Handles job queuing with Upstash Redis
"""

import redis.asyncio as redis
from typing import Optional, Dict, Any, List
from functools import lru_cache
import json
import logging
import uuid

from api.config import settings

logger = logging.getLogger(__name__)


class RedisQueue:
    """Async Redis queue for job management."""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
    
    @property
    async def client(self) -> Optional[redis.Redis]:
        """Get or create Redis client with fallback."""
        if self._client is None:
            if not settings.REDIS_URL:
                return None
            try:
                self._client = redis.from_url(
                    settings.REDIS_URL,
                    password=settings.REDIS_TOKEN if settings.REDIS_TOKEN else None,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
                self._client = None
        return self._client
    
    async def ping(self) -> bool:
        """Test Redis connection."""
        client = await self.client
        try:
            return await client.ping()
        except Exception:
            return False
    
    # ==================== Job Queue Operations ====================
    
    async def enqueue_job(
        self,
        queue_name: str,
        job_data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Enqueue job to a FIFO list. Key matches dequeue_job: queue:{queue_name}."""
        client = await self.client
        if client:
            try:
                job_id = job_data.get("job_id", str(uuid.uuid4()))
                queue_key = f"queue:{queue_name}"
                await client.rpush(queue_key, json.dumps(job_data))
                return job_id
            except Exception as e:
                logger.warning(f"Redis enqueue failed: {e}. Using fallback.")

        logger.info(f"Job queued locally (fallback): {job_data.get('job_id')}")
        return job_data.get("job_id", str(uuid.uuid4()))
    
    async def dequeue_job(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Pop the next job from the FIFO list (LPOP). Matches enqueue_job key."""
        client = await self.client
        if not client:
            return None
        queue_key = f"queue:{queue_name}"
        raw = await client.lpop(queue_key)
        if raw:
            return json.loads(raw)
        return None
    
    async def get_queue_length(self, queue_name: str) -> int:
        """Get the number of jobs in a queue."""
        client = await self.client
        queue_key = f"queue:{queue_name}"
        return await client.zcard(queue_key)
    
    async def get_queue_position(self, queue_name: str, job_id: str) -> Optional[int]:
        """Get the position of a job in the queue (0-indexed)."""
        client = await self.client
        queue_key = f"queue:{queue_name}"
        
        rank = await client.zrank(queue_key, job_id)
        return rank
    
    # ==================== Job Status Operations ====================
    
    async def set_job_status(
        self,
        job_id: str,
        status: str,
        ttl: int = 86400
    ) -> None:
        """
        Set job status with expiration.
        
        Args:
            job_id: Job ID
            status: Status string
            ttl: Time to live in seconds (default 24 hours)
        """
        client = await self.client
        status_key = f"status:{job_id}"
        await client.setex(status_key, ttl, status)
    
    async def get_job_status(self, job_id: str) -> Optional[str]:
        """Get job status."""
        client = await self.client
        status_key = f"status:{job_id}"
        return await client.get(status_key)
    
    # ==================== Pub/Sub for Real-time Updates ====================
    
    async def publish_job_update(
        self,
        job_id: str,
        update_data: Dict[str, Any]
    ) -> None:
        """Publish job update for WebSocket subscribers."""
        client = await self.client
        channel = f"updates:{job_id}"
        await client.publish(channel, json.dumps(update_data))
    
    async def subscribe_to_job_updates(
        self,
        job_id: str
    ) -> redis.client.PubSub:
        """Subscribe to job updates channel."""
        client = await self.client
        channel = f"updates:{job_id}"
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        return pubsub
    
    # ==================== Rate Limiting ====================
    
    async def check_rate_limit(
        self,
        user_id: str,
        action: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if user is within rate limit.
        
        Args:
            user_id: User ID
            action: Action name (e.g., "generate")
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        client = await self.client
        key = f"ratelimit:{user_id}:{action}"
        
        current = await client.get(key)
        
        if current is None:
            await client.setex(key, window_seconds, 1)
            return True, max_requests - 1
        
        current = int(current)
        
        if current >= max_requests:
            return False, 0
        
        await client.incr(key)
        return True, max_requests - current - 1
    
    # ==================== Caching ====================
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        client = await self.client
        value = await client.get(f"cache:{key}")
        return json.loads(value) if value else None
    
    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> None:
        """Set cached value with TTL."""
        client = await self.client
        await client.setex(f"cache:{key}", ttl, json.dumps(value))
    
    async def cache_delete(self, key: str) -> None:
        """Delete cached value."""
        client = await self.client
        await client.delete(f"cache:{key}")
    
    # ==================== GPU FIFO Queue (concurrency=1) ====================

    async def enqueue_gpu_job(self, job_data: Dict[str, Any]) -> str:
        """Add job to GPU FIFO queue (max concurrency=1)."""
        client = await self.client
        job_id = job_data.get("job_id")
        queue_key = "gpu:queue"

        await client.set(f"gpu:job:{job_id}", json.dumps(job_data))
        await client.rpush(queue_key, job_id)

        return job_id

    async def dequeue_gpu_job(self) -> Optional[Dict[str, Any]]:
        """Get next job from GPU FIFO queue."""
        client = await self.client
        queue_key = "gpu:queue"

        job_id = await client.lpop(queue_key)
        if not job_id:
            return None

        job_data = await client.get(f"gpu:job:{job_id}")
        if job_data:
            return json.loads(job_data)
        return None

    async def get_gpu_queue_position(self, job_id: str) -> Optional[int]:
        """Get position in GPU queue (0 = next)."""
        client = await self.client
        queue_key = "gpu:queue"
        return await client.lpos(queue_key, job_id)

    async def get_gpu_queue_length(self) -> int:
        """Get GPU queue length."""
        client = await self.client
        return await client.llen("gpu:queue")

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()


@lru_cache()
def get_redis_queue() -> RedisQueue:
    """Get cached Redis queue instance."""
    return RedisQueue()
