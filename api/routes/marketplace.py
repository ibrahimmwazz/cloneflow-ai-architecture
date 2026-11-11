from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
from api.services.supabase_client import get_supabase, SupabaseClient
from api.dependencies.auth import require_user

router = APIRouter()

class PurchaseRequest(BaseModel):
    price_credits: int

@router.get("/items")
async def list_marketplace_items(
    supabase: SupabaseClient = Depends(get_supabase)
) -> List[Dict[str, Any]]:
    """Get all marketplace items."""
    return await supabase.get_marketplace_items()

@router.post("/items/{item_id}/purchase")
async def purchase_item(
    item_id: str,
    request: PurchaseRequest,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """Purchase a marketplace item using credits."""
    success = await supabase.purchase_marketplace_item(user_id, item_id, request.price_credits)
    if not success:
        raise HTTPException(status_code=400, detail="Insufficient credits or purchase failed")
    return {"message": "Purchase successful"}
