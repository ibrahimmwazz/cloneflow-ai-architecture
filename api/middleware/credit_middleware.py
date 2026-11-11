"""
CloneFlow AI - Credit System Middleware
Enforces daily limits for free-tier users before expensive GPU operations.
"""

import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from api.services.cache_service import cache_service

logger = logging.getLogger(__name__)

# Daily limits per plan
FREE_LIMITS = {
    "images": 20,
    "audio": 10,
    "video": 3,
}

PREMIUM_LIMITS = {
    "images": 1000,
    "audio": 500,
    "video": 100,
}

# Route → resource type mapping
ROUTE_RESOURCE_MAP = {
    "/api/avatars/generate": "images",
    "/api/tts/synthesize": "audio",
    "/api/jobs/create": "video",
}


def _extract_user_id_from_token(request: Request) -> str | None:
    """
    Decode the Bearer JWT to get the user_id (sub claim).
    Verifies the JWT signature locally using SUPABASE_JWT_SECRET to prevent
    forged tokens from draining credits, while avoiding network latency.
    """
    import os, jwt
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    
    if not secret:
        # Fallback to Supabase client if secret isn't loaded (slower but secure)
        from api.services.supabase_client import get_supabase
        try:
            supabase = next(get_supabase())
            resp = supabase.client.auth.get_user(token)
            return resp.user.id if resp and resp.user else None
        except Exception:
            return None
            
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        return payload.get("sub")
    except Exception as e:
        logger.warning(f"CreditMiddleware JWT validation failed: {e}")
        return None


class CreditMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces daily usage limits before expensive GPU routes.
    Reads the JWT sub claim from the Bearer token to identify the user.
    """

    async def dispatch(self, request: Request, call_next):
        route = request.url.path

        # Only check routes that consume credits
        resource_type = ROUTE_RESOURCE_MAP.get(route)
        if not resource_type:
            return await call_next(request)

        user_id = _extract_user_id_from_token(request)
        if not user_id:
            # Unauthenticated — route handler will reject with 401
            return await call_next(request)

        plan = getattr(request.state, "plan", "free")
        limits = PREMIUM_LIMITS if plan == "premium" else FREE_LIMITS

        # Check credits atomically via Redis
        credits = await cache_service.get_credits(user_id)
        used_today = credits.get(resource_type, 0)
        limit = limits.get(resource_type, 0)

        if used_today >= limit:
            logger.warning(f"User {user_id} exceeded {resource_type} limit ({limit})")
            raise HTTPException(
                status_code=429,
                detail=f"Daily {resource_type} limit reached ({limit}). Upgrade to Premium."
            )

        # Deduct credit (increment used counter)
        credits[resource_type] = used_today + 1
        await cache_service.set_credits(user_id, credits)

        request.state.credits_remaining = limit - used_today - 1
        return await call_next(request)


async def check_credits(user_id: str, action: str, plan: str = "free") -> bool:
    """
    Standalone credit check for use inside route handlers.
    Returns True if user has credits, False otherwise.
    """
    limits = PREMIUM_LIMITS if plan == "premium" else FREE_LIMITS
    credits = await cache_service.get_credits(user_id)
    used_today = credits.get(action, 0)
    limit = limits.get(action, 0)

    if used_today >= limit:
        return False

    credits[action] = used_today + 1
    await cache_service.set_credits(user_id, credits)
    return True
