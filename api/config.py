"""
CloneFlow AI - Configuration Settings
Environment-based configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "CloneFlow AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Base URL of the API for worker webhooks (e.g. Koyeb/Fly.io URL)
    API_BASE_URL: str = "https://ai-clone.fly.dev"
    
    # CORS (hardcoded in main.py, this is fallback only)
    CORS_ORIGINS: str = ""

    # Public frontend URL (used for Stripe redirect links)
    APP_URL: str = "https://ai-clone-tan-six.vercel.app"

    # Shared secret between Fly.io API and Modal worker.
    # Set the same value in both Fly.io env vars and the Modal cloneflow-fly secret.
    # Protects the /api/jobs/{id}/update and /api/jobs/queued/list endpoints.
    WORKER_SECRET: str = ""
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    
    # RunPod (Deprecated - using Modal instead)
    RUNPOD_API_KEY: str = ""
    RUNPOD_ENDPOINT_ID: str = ""
    RUNPOD_WEBHOOK_SECRET: str = ""

    # Lightning AI (Free GPU — 80 hrs/month)
    LIGHTNING_AI_API_KEY: str = ""
    LIGHTNING_AI_BASE_URL: str = ""

    # Redis (Upstash)
    REDIS_URL: str = ""
    REDIS_TOKEN: str = ""

    # LLMs, RAG & Embeddings
    GROQ_API_KEY: str = ""
    HF_TOKEN: str = ""
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RAG_TOP_K: int = 4
    
    # Cloudflare R2
    R2_ACCOUNT_ID: str
    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_BUCKET_NAME: str = "cloneflow-videos"
    R2_PUBLIC_URL: str
    
    # ElevenLabs (Voice Cloning)
    ELEVENLABS_API_KEY: str = ""
    
    # Stripe
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_ID_CREDITS: str = ""
    
    # C2PA Compliance
    C2PA_PRIVATE_KEY_PATH: str = "./keys/c2pa_private.pem"
    C2PA_PUBLIC_KEY_PATH: str = "./keys/c2pa_public.pem"
    
    # Watermark
    WATERMARK_PATH: str = "./assets/cr_icon.png"
    
    # Sentry Error Monitoring
    SENTRY_DSN: str = ""
    
    # Email Service (Resend)
    RESEND_API_KEY: str = ""
    EMAIL_FROM_ADDRESS: str = "noreply@cloneflow.ai"
    EMAIL_FROM_NAME: str = "CloneFlow AI"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()
