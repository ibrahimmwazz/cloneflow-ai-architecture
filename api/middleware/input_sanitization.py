"""
CloneFlow AI - Input Sanitization Middleware
Sanitizes user input to prevent XSS and injection attacks
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import re
import logging

logger = logging.getLogger(__name__)


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Middleware for sanitizing user input."""
    
    # Patterns for detecting potential attacks
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
    ]
    
    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(\bexec\b|\bexecute\b)",
        r"(--|#|\/\*|\*\/)",
        r"(\bor\s+1\s*=\s*1)",
        r"(\band\s+1\s*=\s*1)",
    ]
    
    def _check_for_attack_patterns(self, value: str) -> bool:
        """Check if value contains attack patterns."""
        if not isinstance(value, str):
            return False
        
        # Check for XSS patterns
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"XSS pattern detected: {pattern}")
                return True
        
        # Check for SQL injection patterns
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return True
        
        return False
    
    def _sanitize_dict(self, data: dict) -> dict:
        """Recursively sanitize dictionary values."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = self._sanitize_list(value)
            elif isinstance(value, str):
                if self._check_for_attack_patterns(value):
                    raise HTTPException(
                        status_code=400,
                        detail="Input contains potentially malicious content"
                    )
                # Basic sanitization - remove HTML tags
                sanitized[key] = re.sub(r'<[^>]+>', '', value)
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_list(self, data: list) -> list:
        """Recursively sanitize list values."""
        sanitized = []
        for item in data:
            if isinstance(item, dict):
                sanitized.append(self._sanitize_dict(item))
            elif isinstance(item, list):
                sanitized.append(self._sanitize_list(item))
            elif isinstance(item, str):
                if self._check_for_attack_patterns(item):
                    raise HTTPException(
                        status_code=400,
                        detail="Input contains potentially malicious content"
                    )
                sanitized.append(re.sub(r'<[^>]+>', '', item))
            else:
                sanitized.append(item)
        return sanitized
    
    async def dispatch(self, request: Request, call_next):
        """Sanitize request body if it's JSON."""
        
        # Only sanitize JSON requests
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
                
                # Sanitize the body
                if isinstance(body, dict):
                    sanitized_body = self._sanitize_dict(body)
                    # Replace the request body with sanitized version
                    # Note: This is a simplified approach - in production, you'd want
                    # to use a more sophisticated method to modify the request
                    import json
                    request._body = json.dumps(sanitized_body).encode("utf-8")
                    
            except Exception as e:
                logger.error(f"Error during input sanitization: {e}")
                # Don't block the request if sanitization fails
                # Let the endpoint handle invalid data
        
        response = await call_next(request)
        return response
