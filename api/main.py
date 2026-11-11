"""
CloneFlow AI - FastAPI Backend Entry Point
Serverless AI Avatar Clone SaaS for content creators
"""


from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import logging
import traceback
import sentry_sdk

from api.routes import jobs, webhooks, billing, auth, avatars, tts, captions, scripts, video_editor, screen_recording, collaboration, scorm, streaming, gpu_optimization, cache, monitoring, multi_avatar, upload_simple, ai_tools, video_gen, dubbing, image_gen, realtime, admin, analytics, marketplace, rag_agent
from api.services.supabase_client import get_supabase
from api.services.redis_queue import get_redis_queue
from api.middleware.credit_middleware import CreditMiddleware
from api.middleware.request_validation import RequestValidationMiddleware
from api.middleware.input_sanitization import InputSanitizationMiddleware
from api.utils.error_responses import ErrorResponse, APIError
from api.config import settings
import os

from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Sentry for error monitoring
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        environment=os.getenv("ENVIRONMENT", "development"),
    )
    logger.info("Sentry initialized for error monitoring")
else:
    logger.info("Sentry not configured - SENTRY_DSN not set")


async def cleanup_loop(supabase):
    """Background task to clean up jobs stuck in processing."""
    import asyncio
    while True:
        try:
            count = await supabase.cleanup_zombie_jobs()
            if count > 0:
                logger.info(f"Cleaned up {count} zombie jobs")
        except Exception as e:
            logger.error(f"Zombie cleanup failed: {e}")
        await asyncio.sleep(300)  # Run every 5 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    logger.info("🚀 CloneFlow AI Backend starting up...")
    supabase = get_supabase()
    redis = get_redis_queue()
    
    import asyncio
    cleanup_task = asyncio.create_task(cleanup_loop(supabase))
    
    modal_url = os.getenv("MODAL_WORKER_URL", "")
    if modal_url:
        logger.info("✅ Modal worker configured — GPU jobs dispatched to Modal")
    else:
        logger.warning("⚠️  MODAL_WORKER_URL not set — GPU jobs will queue in Redis")
    logger.info("✅ All services initialized")
    
    yield
    
    # Cleanup
    cleanup_task.cancel()
    logger.info("🔌 CloneFlow AI Backend shutting down...")


# Rate limiter — 60 req/min per IP by default, stricter on heavy endpoints
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="CloneFlow AI",
    description="Serverless AI Avatar Clone SaaS API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Trust reverse proxy X-Forwarded-* headers (Fly-Proxy HTTPS enforcement)
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-clone-tan-six.vercel.app",
        "https://ai-clone.fly.dev",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Worker-Secret", "Range"],
    expose_headers=["Content-Type", "X-Accel-Buffering", "Content-Range", "Accept-Ranges", "Content-Length"],
)

# Request validation middleware (security)
app.add_middleware(RequestValidationMiddleware)

# Input sanitization middleware (security)
app.add_middleware(InputSanitizationMiddleware)

# IP-based rate limiting (slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Credit-based rate limiting for free-tier users
app.add_middleware(CreditMiddleware)

# Global exception handler - ensures CORS headers on ALL error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions with standardized error responses."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Generate request ID for tracking
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    error_response = ErrorResponse.create(
        message="An unexpected error occurred",
        status_code=500,
        error_code="INTERNAL_SERVER_ERROR",
        details={"exception": str(exc)} if settings.DEBUG else None,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handle custom API errors with standardized responses."""
    logger.warning(f"API Error: {exc.error_code} - {exc.message}")
    
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    error_response = ErrorResponse.create(
        message=exc.message,
        status_code=exc.status_code,
        error_code=exc.error_code,
        details=exc.details if settings.DEBUG else None,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["Webhooks"])
app.include_router(billing.router, prefix="/api/billing", tags=["Billing"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(marketplace.router, prefix="/api/marketplace", tags=["Marketplace"])
app.include_router(avatars.router, prefix="/api", tags=["Avatars"])
app.include_router(tts.router, prefix="/api", tags=["TTS"])
app.include_router(captions.router, prefix="/api", tags=["Captions"])
app.include_router(scripts.router, prefix="/api", tags=["Scripts"])
app.include_router(video_editor.router, prefix="/api", tags=["Video Editor"])
app.include_router(screen_recording.router, prefix="/api", tags=["Screen Recording"])
app.include_router(collaboration.router, prefix="/api", tags=["Collaboration"])
app.include_router(scorm.router, prefix="/api", tags=["SCORM"])
app.include_router(streaming.router, prefix="/api", tags=["Streaming"])
app.include_router(gpu_optimization.router, prefix="/api", tags=["GPU Optimization"])
app.include_router(cache.router, prefix="/api", tags=["Cache"])
app.include_router(monitoring.router, prefix="/api", tags=["Monitoring"])
app.include_router(multi_avatar.router, prefix="/api", tags=["Multi-Avatar"])
app.include_router(upload_simple.router, prefix="/api", tags=["Upload"])
app.include_router(ai_tools.router, prefix="/api", tags=["AI Tools"])
app.include_router(video_gen.router, prefix="/api", tags=["Video Generation"])
app.include_router(dubbing.router, prefix="/api", tags=["Dubbing"])
app.include_router(image_gen.router, prefix="/api", tags=["Image Generation"])
app.include_router(realtime.router, prefix="/api", tags=["Real-time Streaming"])
app.include_router(rag_agent.router, prefix="/api", tags=["RAG & AI Agents"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CloneFlow AI",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check for all services."""
    import httpx
    health_status = {
        "api":            "healthy",
        "supabase":       "unknown",
        "redis":          "unknown",
        "modal_avatar":   "not_configured",
        "modal_imggen":   "not_configured",
        "modal_vidgen":   "not_configured",
    }

    try:
        supabase = get_supabase()
        health_status["supabase"] = "healthy"
    except Exception as e:
        health_status["supabase"] = f"error: {str(e)}"

    try:
        redis = get_redis_queue()
        await redis.ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"

    # Modal worker health (lightweight — just checks if URL is configured)
    for key, env_var in [
        ("modal_avatar", "MODAL_WORKER_URL"),
        ("modal_imggen", "MODAL_IMGGEN_URL"),
        ("modal_vidgen", "MODAL_VIDGEN_URL"),
    ]:
        url = os.getenv(env_var, "")
        if url:
            health_status[key] = "configured"

    return health_status


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
