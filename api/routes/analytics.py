from fastapi import APIRouter, Depends
from typing import Dict, Any
from api.services.supabase_client import get_supabase, SupabaseClient
from api.dependencies.auth import require_user

router = APIRouter()

@router.get("/stats")
async def get_analytics_stats(
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
) -> Dict[str, Any]:
    """Get user analytics statistics."""
    stats = await supabase.get_analytics_stats(user_id)
    return stats
