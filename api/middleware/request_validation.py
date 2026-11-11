"""
CloneFlow AI - Request Validation Middleware
Validates request size, content-type, and other security checks
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for validating incoming requests."""
    
    # Maximum request body size (10MB)
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Allowed content types
    ALLOWED_CONTENT_TYPES = [
        "application/json",
        "multipart/form-data",
        "application/x-www-form-urlencoded",
        "text/plain",
    ]
    
    async def dispatch(self, request: Request, call_next):
        """Validate request before processing."""
        
        # Check request size for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    if size > self.MAX_REQUEST_SIZE:
                        logger.warning(f"Request too large: {size} bytes")
                        raise HTTPException(
                            status_code=413,
                            detail=f"Request too large. Maximum size is {self.MAX_REQUEST_SIZE / (1024*1024)}MB"
                        )
                except ValueError:
                    pass
        
        # Validate content-type for requests with body
        if request.method in ["POST", "PUT", "PATCH"] and request.headers.get("content-type"):
            content_type = request.headers.get("content-type").split(";")[0].strip()
            if content_type not in self.ALLOWED_CONTENT_TYPES:
                logger.warning(f"Invalid content-type: {content_type}")
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported media type: {content_type}"
                )
        
        # Check for suspicious patterns in headers
        user_agent = request.headers.get("user-agent", "")
        if not user_agent or len(user_agent) < 10:
            logger.warning(f"Suspicious request with missing/short user-agent from {request.client.host}")
            # Don't block, just log - some legitimate clients may not send UA
        
        response = await call_next(request)
        return response
