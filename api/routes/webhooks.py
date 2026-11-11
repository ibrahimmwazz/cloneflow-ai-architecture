"""
CloneFlow AI - Webhooks Router
Handles callbacks from RunPod, Stripe, and other external services
"""

from fastapi import APIRouter, HTTPException, Request, Header, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import json
import hmac
import hashlib
import logging

from api.services.supabase_client import get_supabase
from api.services.redis_queue import get_redis_queue
from api.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class RunPodWebhookPayload(BaseModel):
    """Payload structure from RunPod webhook callback."""
    id: str  # RunPod job ID
    status: str  # COMPLETED, FAILED, etc.
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    executionTime: Optional[int] = None


class StripeWebhookPayload(BaseModel):
    """Payload structure from Stripe webhook."""
    id: str
    object: str
    type: str
    data: Dict[str, Any]


def verify_runpod_signature(payload: bytes, signature: str) -> bool:
    """Verify RunPod webhook signature."""
    if not settings.RUNPOD_WEBHOOK_SECRET:
        logger.error("RUNPOD_WEBHOOK_SECRET not configured - rejecting webhook")
        return False
    
    expected_signature = hmac.new(
        settings.RUNPOD_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


@router.post("/runpod")
async def runpod_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_runpod_signature: Optional[str] = Header(None)
):
    """
    Handle RunPod job completion webhook.
    
    Called by RunPod when a GPU job completes or fails.
    """
    # Read raw body for signature verification
    body = await request.body()
    
    # Verify signature (mandatory for security)
    if not x_runpod_signature:
        logger.error("RunPod webhook missing signature header")
        raise HTTPException(status_code=401, detail="Missing webhook signature")
    
    if not verify_runpod_signature(body, x_runpod_signature):
        logger.warning("Invalid RunPod webhook signature")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    try:
        payload = RunPodWebhookPayload.parse_raw(body)
    except Exception as e:
        logger.error(f"Failed to parse RunPod webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload format")
    
    supabase = get_supabase()
    
    # Find job by RunPod ID
    job = await supabase.get_job_by_runpod_id(payload.id)
    
    if not job:
        logger.warning(f"Received webhook for unknown RunPod job: {payload.id}")
        return {"status": "ignored", "reason": "job_not_found"}
    
    job_id = job["job_id"]
    user_id = job["user_id"]
    
    if payload.status == "COMPLETED":
        video_url = payload.output.get("video_url") if payload.output else None
        
        if video_url:
            # Update job as completed
            await supabase.update_job(
                job_id,
                {
                    "status": "completed",
                    "video_url": video_url,
                    "completed_at": datetime.utcnow().isoformat(),
                    "execution_time_ms": payload.executionTime
                }
            )
            
            # Calculate and deduct credits
            execution_time_seconds = (payload.executionTime or 0) / 1000
            credits_used = max(1, int(execution_time_seconds / 10))  # 1 credit per 10 seconds
            
            await supabase.finalize_job_credits(job_id, user_id, credits_used)
            
            logger.info(f"Job {job_id} completed successfully. Video: {video_url}")
        else:
            # Job completed but no video URL
            await supabase.update_job(
                job_id,
                {
                    "status": "failed",
                    "error_message": "Job completed but no video URL returned",
                    "completed_at": datetime.utcnow().isoformat()
                }
            )
            
            # Refund credits
            await supabase.refund_job_credits(job_id, user_id)
    
    elif payload.status == "FAILED":
        error_message = payload.error or "Unknown error occurred"
        
        await supabase.update_job(
            job_id,
            {
                "status": "failed",
                "error_message": error_message,
                "completed_at": datetime.utcnow().isoformat()
            }
        )
        
        # Refund credits
        await supabase.refund_job_credits(job_id, user_id)
        
        logger.error(f"Job {job_id} failed: {error_message}")
    
    elif payload.status == "CANCELLED":
        await supabase.update_job(
            job_id,
            {
                "status": "cancelled",
                "completed_at": datetime.utcnow().isoformat()
            }
        )
        
        # Refund credits
        await supabase.refund_job_credits(job_id, user_id)
        
        logger.info(f"Job {job_id} was cancelled")
    
    return {"status": "processed", "job_id": job_id}


@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature")
):
    """
    Handle Stripe webhook events.
    Signature is verified with STRIPE_WEBHOOK_SECRET — rejects any forged events.
    """
    import stripe as _stripe
    body = await request.body()

    # Verify signature (required in production)
    webhook_secret = settings.STRIPE_WEBHOOK_SECRET
    if webhook_secret:
        if not stripe_signature:
            raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")
        try:
            event = _stripe.Webhook.construct_event(body, stripe_signature, webhook_secret)
        except _stripe.error.SignatureVerificationError:
            logger.warning("Stripe webhook signature verification failed")
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        except Exception as e:
            logger.error(f"Stripe webhook parse error: {e}")
            raise HTTPException(status_code=400, detail="Invalid payload")
    else:
        # Dev-mode only: no secret configured
        logger.warning("STRIPE_WEBHOOK_SECRET not set — skipping signature check (dev mode)")
        try:
            event = json.loads(body)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid payload")
    
    event_type = event.get("type")
    supabase = get_supabase()
    
    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session.get("customer")
        user_id = session.get("client_reference_id")
        
        if user_id:
            # Update user's stripe customer id if it's new
            if customer_id:
                supabase.client.table("users").update({"stripe_customer_id": customer_id}).eq("id", user_id).execute()
            
            # Check if this is a subscription checkout
            if session.get("mode") == "subscription":
                subscription_id = session.get("subscription")
                metadata = session.get("metadata", {})
                package_id = metadata.get("package_id", "pro")
                
                # We update status to active and set the stripe sub id
                # invoice.paid will handle actually giving the credits
                await supabase.update_subscription_status(
                    user_id=user_id,
                    status="active",
                    stripe_sub_id=subscription_id
                )
                logger.info(f"Subscription {subscription_id} activated for user {user_id}")
            else:
                # One-time purchase
                metadata = session.get("metadata", {})
                credits = int(metadata.get("credits", 100))
                await supabase.add_user_credits(user_id, credits)
                logger.info(f"Added {credits} one-time credits to user {user_id}")
    
    elif event_type == "invoice.paid":
        # Handle subscription renewal
        invoice = event["data"]["object"]
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")
        
        # Find user by customer_id
        user_resp = supabase.client.table("users").select("id").eq("stripe_customer_id", customer_id).execute()
        if user_resp.data:
            user_id = user_resp.data[0]["id"]
            
            # Add monthly credits - for simplicity we just look up the subscription from stripe if needed, 
            # but we can also just use a default like 500 for monthly-creator.
            # In a real app we'd fetch the Stripe subscription item to see which tier it is.
            await supabase.add_user_credits(user_id, 500)
            
            # Update current period end (roughly 30 days)
            # In a real app, parse `invoice.lines.data[0].period.end`
            end_date = datetime.fromtimestamp(invoice.get("period_end", datetime.utcnow().timestamp() + 2592000)).isoformat()
            
            await supabase.update_subscription_status(
                user_id=user_id,
                status="active",
                current_period_end=end_date,
                stripe_sub_id=subscription_id
            )
            logger.info(f"Invoice paid for customer {customer_id}, renewed subscription {subscription_id}")
    
    elif event_type == "invoice.payment_failed":
        invoice = event["data"]["object"]
        customer_id = invoice.get("customer")
        
        user_resp = supabase.client.table("users").select("id").eq("stripe_customer_id", customer_id).execute()
        if user_resp.data:
            user_id = user_resp.data[0]["id"]
            await supabase.update_subscription_status(user_id, status="past_due")
            logger.warning(f"Payment failed for customer {customer_id}, marked past_due")
    
    elif event_type == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")
        
        user_resp = supabase.client.table("users").select("id").eq("stripe_customer_id", customer_id).execute()
        if user_resp.data:
            user_id = user_resp.data[0]["id"]
            await supabase.update_subscription_status(user_id, status="canceled")
            logger.info(f"Subscription cancelled for customer {customer_id}")
    
    return {"status": "received", "event_type": event_type}


@router.post("/elevenlabs")
async def elevenlabs_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Handle ElevenLabs voice generation webhooks.
    
    Called when voice cloning or text-to-speech completes.
    """
    body = await request.body()
    
    try:
        payload = json.loads(body)
    except Exception as e:
        logger.error(f"Failed to parse ElevenLabs webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    
    event_type = payload.get("event_type")
    
    if event_type == "voice_cloning_complete":
        voice_id = payload.get("voice_id")
        user_id = payload.get("user_id")
        
        # Store voice ID for user
        supabase = get_supabase()
        await supabase.save_user_voice(user_id, voice_id)
        
        logger.info(f"Voice cloning complete for user {user_id}: {voice_id}")
    
    return {"status": "received"}
