"""
CloneFlow AI - Real-time Job Status Streaming
Server-Sent Events (SSE) endpoint for real-time job status updates
Eliminates HTTP polling by using Redis Pub/Sub
"""

import json
import logging
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from api.services.redis_queue import get_redis_queue, RedisQueue
from api.services.supabase_client import get_supabase, SupabaseClient
from api.dependencies.auth import require_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/realtime", tags=["Real-time Streaming"])


class JobStatusEvent(BaseModel):
    """Job status update event structure."""
    job_id: str
    status: str
    video_url: str | None = None
    image_url: str | None = None
    error_message: str | None = None
    progress: int | None = None
    timestamp: str


async def _job_status_event_generator(
    job_id: str,
    user_id: str,
    redis: RedisQueue,
    supabase: SupabaseClient,
) -> AsyncGenerator[dict, None]:
    """
    Generator function that yields SSE events for job status updates.
    
    Subscribes to Redis Pub/Sub channel 'job_status:{job_id}' and yields
    messages as they arrive. Also handles client disconnects gracefully.
    
    Args:
        job_id: The job ID to stream updates for
        user_id: The authenticated user ID (for ownership verification)
        redis: Redis queue client
        supabase: Supabase client for initial status fetch
        
    Yields:
        dict: SSE event data with 'event', 'data' keys
    """
    channel_name = f"job_status:{job_id}"
    pubsub = None
    
    try:
        # Verify user owns this job
        job = await supabase.get_job(job_id, user_id)
        if not job:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Job not found or access denied"})
            }
            return
        
        # Send initial status
        yield {
            "event": "status",
            "data": json.dumps({
                "job_id": job_id,
                "status": job["status"],
                "video_url": job.get("video_url"),
                "image_url": job.get("image_url"),
                "error_message": job.get("error_message"),
                "progress": job.get("progress"),
                "timestamp": job.get("updated_at", job.get("created_at"))
            })
        }
        
        # If job is already completed or failed, no need to stream further
        if job["status"] in ["completed", "failed", "cancelled"]:
            yield {
                "event": "done",
                "data": json.dumps({
                    "job_id": job_id,
                    "status": job["status"],
                    "message": f"Job is {job['status']}, streaming complete"
                })
            }
            return
        
        # Subscribe to Redis Pub/Sub channel
        client = await redis.client
        if not client:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Redis not available"})
            }
            return
        
        pubsub = client.pubsub()
        await pubsub.subscribe(channel_name)
        logger.info(f"User {user_id[:8]} subscribed to {channel_name}")
        
        # Listen for messages — heartbeat every 15s, status on Redis message
        last_heartbeat = asyncio.get_event_loop().time()
        HEARTBEAT_INTERVAL = 15.0  # seconds between keep-alive pings
        MAX_IDLE_SECONDS = 200     # kill stream if no update for 200s (Modal 3-min cap + buffer)
        last_message_time = asyncio.get_event_loop().time()

        while True:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(timeout=1.0),
                    timeout=1.0
                )

                now = asyncio.get_event_loop().time()

                if message and message["type"] == "message":
                    last_message_time = now
                    try:
                        data = json.loads(message["data"])

                        if data.get("job_id") == job_id:
                            yield {
                                "event": "status",
                                "data": json.dumps(data)
                            }

                            if data.get("status") in ["completed", "failed", "cancelled"]:
                                yield {
                                    "event": "done",
                                    "data": json.dumps({
                                        "job_id": job_id,
                                        "status": data.get("status"),
                                        "message": f"Job {data.get('status')}, streaming complete"
                                    })
                                }
                                break
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Redis message: {e}")

                # Send heartbeat on interval only (not on every poll)
                if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                    last_heartbeat = now
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"ts": int(now)})
                    }

                # Kill stream if job is silently stuck beyond Modal timeout window
                if now - last_message_time > MAX_IDLE_SECONDS:
                    logger.warning(f"SSE stream idle {MAX_IDLE_SECONDS}s for job {job_id[:8]} — force-closing")
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Job timed out — no update received"})
                    }
                    break

            except asyncio.TimeoutError:
                pass  # Normal — poll interval, no message yet
                
    except asyncio.CancelledError:
        logger.info(f"SSE stream cancelled for job {job_id[:8]} by user {user_id[:8]}")
        raise
        
    except Exception as e:
        logger.error(f"Error in SSE stream for job {job_id[:8]}: {e}")
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)})
        }
        
    finally:
        # Clean up Redis subscription
        if pubsub:
            try:
                await pubsub.unsubscribe(channel_name)
                await pubsub.close()
                logger.info(f"Unsubscribed from {channel_name}")
            except Exception as e:
                logger.error(f"Error closing Redis pubsub: {e}")


@router.get("/jobs/{job_id}")
async def stream_job_status(
    job_id: str,
    user_id: str = Depends(require_user),
    redis: RedisQueue = Depends(get_redis_queue),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Server-Sent Events endpoint for real-time job status updates.
    
    This endpoint streams job status updates in real-time using SSE,
    eliminating the need for HTTP polling.
    
    Usage:
        const eventSource = new EventSource('/api/realtime/jobs/{job_id}');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Job status:', data.status);
        };
        eventSource.addEventListener('done', () => {
            eventSource.close();
        });
    
    Events:
        - status: Job status update (processing, completed, failed, etc.)
        - heartbeat: Keep-alive signal (every 5 seconds)
        - error: Error occurred
        - done: Job reached terminal state, streaming complete
    
    Args:
        job_id: The job ID to stream updates for
        user_id: Authenticated user ID (injected by require_user)
        redis: Redis queue client (injected)
        supabase: Supabase client (injected)
        
    Returns:
        EventSourceResponse: SSE stream
    """
    return EventSourceResponse(
        _job_status_event_generator(job_id, user_id, redis, supabase),
        ping=15,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
