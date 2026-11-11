"""
CloneFlow AI - Auth Dependencies
FastAPI Depends() helpers that validate Supabase JWT tokens from
the Authorization: Bearer <token> header.
"""

import logging
from fastapi import Request, HTTPException, Depends
from typing import Optional

from api.services.supabase_client import get_supabase, SupabaseClient

logger = logging.getLogger(__name__)


def require_user(
    request: Request,
    supabase: SupabaseClient = Depends(get_supabase),
) -> str:
    """
    Validate Bearer token and return the authenticated user_id.
    Accepts the token via:
      1. Authorization: Bearer <token> header (standard REST calls)
      2. ?token=<token> query parameter (required for EventSource/SSE connections
         because the browser's native EventSource API cannot send custom headers)
    Raises HTTP 401 if the token is missing or invalid.
    """
    # 1. Try Authorization header first (preferred for non-SSE requests)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        # 2. Fall back to ?token= query param (for SSE/EventSource clients)
        token = request.query_params.get("token", "")
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        resp = supabase.client.auth.get_user(token)
        if not resp or not resp.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return resp.user.id
    except HTTPException:
        raise
    except Exception as e:
        logger.debug(f"Token validation failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def optional_user(
    request: Request,
    supabase: SupabaseClient = Depends(get_supabase),
) -> Optional[str]:
    """
    Validate Bearer token if present; return user_id or None.
    Used for endpoints that support both authenticated and public access.
    Raises HTTP 401 if a token is provided but is invalid or forged.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None

    token = auth[7:]
    try:
        resp = supabase.client.auth.get_user(token)
        if resp and resp.user:
            return resp.user.id
    except Exception as e:
        logger.debug(f"Optional token validation failed: {e}")
        # Token was provided but is invalid - do not silently ignore
        raise HTTPException(status_code=401, detail="Invalid or expired token")
        
    return None
