"""
CloneFlow AI - Billing Router
Handles Stripe integration, credit management, and subscription handling
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import stripe

from api.services.supabase_client import get_supabase, SupabaseClient
from api.dependencies.auth import require_user
from api.config import settings

router = APIRouter()

# Configure Stripe
if settings.STRIPE_SECRET_KEY:
    stripe.api_key = settings.STRIPE_SECRET_KEY


class CreditPackage(BaseModel):
    """Available credit package for purchase."""
    id: str
    name: str
    credits: int
    price_cents: int
    price_display: str
    popular: bool = False


class UserCredits(BaseModel):
    """User's current credit balance."""
    user_id: str
    credits: int
    credits_used_this_month: int
    subscription_tier: str = "free"


class CheckoutSessionRequest(BaseModel):
    """Request to create a Stripe checkout session."""
    package_id: str
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class CheckoutSessionResponse(BaseModel):
    """Response with Stripe checkout session URL."""
    session_id: str
    checkout_url: str


# Available credit packages
CREDIT_PACKAGES = [
    CreditPackage(
        id="starter",
        name="Starter Pack",
        credits=100,
        price_cents=1000,  # $10.00
        price_display="$10",
        popular=False
    ),
    CreditPackage(
        id="creator",
        name="Creator Pack",
        credits=500,
        price_cents=4500,  # $45.00 (10% discount)
        price_display="$45",
        popular=True
    ),
    CreditPackage(
        id="pro",
        name="Pro Pack",
        credits=1500,
        price_cents=12000,  # $120.00 (20% discount)
        price_display="$120",
        popular=False
    ),
    CreditPackage(
        id="enterprise",
        name="Enterprise Pack",
        credits=5000,
        price_cents=35000,  # $350.00 (30% discount)
        price_display="$350",
        popular=False
    ),
]

SUBSCRIPTION_PACKAGES = [
    CreditPackage(
        id="monthly-starter",
        name="Starter Monthly",
        credits=100,
        price_cents=900,  # $9.00
        price_display="$9/mo",
        popular=False
    ),
    CreditPackage(
        id="monthly-creator",
        name="Creator Monthly",
        credits=500,
        price_cents=3900,  # $39.00
        price_display="$39/mo",
        popular=True
    ),
    CreditPackage(
        id="monthly-pro",
        name="Pro Monthly",
        credits=1500,
        price_cents=9900,  # $99.00
        price_display="$99/mo",
        popular=False
    ),
]


@router.get("/packages", response_model=List[CreditPackage])
async def get_credit_packages():
    """
    Get available credit packages for purchase.
    
    Each package offers a different number of credits at various price points.
    """
    return CREDIT_PACKAGES


@router.get("/balance", response_model=UserCredits)
async def get_credit_balance(
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Get the current user's credit balance and usage statistics.
    """
    user_data = await supabase.get_user_credits_data(user_id)
    
    if not user_data:
        # Create default user record
        await supabase.create_user_credits(user_id, initial_credits=10)
        user_data = {"credits": 10, "credits_used_this_month": 0, "subscription_tier": "free"}
    
    return UserCredits(
        user_id=user_id,
        credits=user_data.get("credits", 0),
        credits_used_this_month=user_data.get("credits_used_this_month", 0),
        subscription_tier=user_data.get("subscription_tier", "free")
    )


@router.post("/checkout", response_model=CheckoutSessionResponse)
async def create_checkout_session(
    request: CheckoutSessionRequest,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Create a Stripe checkout session for purchasing credits.
    
    Redirects user to Stripe-hosted payment page.
    """
    # Find the package
    package = next((p for p in CREDIT_PACKAGES if p.id == request.package_id), None)
    is_subscription = False
    
    if not package:
        package = next((p for p in SUBSCRIPTION_PACKAGES if p.id == request.package_id), None)
        if package:
            is_subscription = True
    
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
    
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    try:
        # Create Stripe checkout session
        app_url = settings.APP_URL.rstrip("/")
        
        price_data = {
            "currency": "usd",
            "product_data": {
                "name": package.name,
                "description": f"{package.credits} video generation credits per month" if is_subscription else f"{package.credits} video generation credits"
            },
            "unit_amount": package.price_cents
        }
        
        if is_subscription:
            price_data["recurring"] = {"interval": "month"}
            
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": price_data,
                "quantity": 1
            }],
            mode="subscription" if is_subscription else "payment",
            success_url=request.success_url or f"{app_url}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=request.cancel_url or f"{app_url}/billing/cancel",
            client_reference_id=user_id,
            metadata={
                "user_id": user_id,
                "package_id": package.id,
                "credits": package.credits
            }
        )
        
        return CheckoutSessionResponse(
            session_id=session.id,
            checkout_url=session.url
        )
    
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/create-portal-session")
async def create_portal_session(
    request: Request,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Create a Stripe billing portal session.
    
    Allows users to manage their payment methods and view invoices.
    """
    if not settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    
    user_data = await supabase.get_user(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    customer_id = user_data.get("stripe_customer_id")
    
    if not customer_id:
        raise HTTPException(status_code=400, detail="No Stripe customer found. Please make a purchase first.")
    
    try:
        app_url = settings.APP_URL.rstrip("/")
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{app_url}/billing"
        )
        return {"portal_url": session.url}
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/add-credits")
async def add_credits_admin(
    user_id: str,
    credits: int,
    x_worker_secret: Optional[str] = Header(None),
    supabase: SupabaseClient = Depends(get_supabase),
):
    """
    Add credits to a user's account.
    Protected by WORKER_SECRET header — only callable by the Modal worker or admin scripts.
    """
    expected = settings.WORKER_SECRET
    if not expected or x_worker_secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    if credits <= 0 or credits > 100000:
        raise HTTPException(status_code=400, detail="Invalid credit amount")
    await supabase.add_user_credits(user_id, credits)
    return {"message": f"Added {credits} credits to user {user_id}"}


@router.get("/transactions")
async def get_transaction_history(
    page: int = 1,
    page_size: int = 20,
    user_id: str = Depends(require_user),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Get the user's transaction history.
    
    Includes credit purchases, usage, and refunds.
    """
    transactions, total = await supabase.get_user_transactions(
        user_id=user_id,
        page=page,
        page_size=page_size
    )
    
    return {
        "transactions": transactions,
        "total": total,
        "page": page,
        "page_size": page_size
    }
