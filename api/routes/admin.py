from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from api.services.supabase_client import get_supabase, SupabaseClient
from api.dependencies.auth import require_user

router = APIRouter()

@router.get("/stats")
async def get_admin_stats(
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
) -> Dict[str, Any]:
    """Get global admin statistics."""
    user = await supabase.get_user(user_id)
    if not user or not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    stats = await supabase.get_admin_stats()
    return stats
