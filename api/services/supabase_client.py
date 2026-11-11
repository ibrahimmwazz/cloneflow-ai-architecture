"""
CloneFlow AI - Supabase Client Service
Handles all database operations and authentication via Supabase
"""

from supabase import create_client, Client
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache
import logging
import httpx

from api.config import settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Wrapper for Supabase client with typed methods."""
    
    def __init__(self):
        self._client: Optional[Client] = None
        self._service_key: str = settings.SUPABASE_SERVICE_ROLE_KEY
    
    @property
    def client(self) -> Client:
        """Get or create Supabase client with connection pooling."""
        from supabase.client import ClientOptions
        
        if self._client is None:
            self._client = create_client(
                settings.SUPABASE_URL,
                self._service_key
            )
        return self._client
    
    def _admin_headers(self) -> dict:
        """Headers that use the service role key directly — always bypasses RLS."""
        return {
            "Authorization": f"Bearer {self._service_key}",
            "apikey": self._service_key,
            "Content-Type": "application/json",
        }

    def _rest_url(self, table: str) -> str:
        return f"{settings.SUPABASE_URL}/rest/v1/{table}"

    def _admin_table(self, table: str):
        """Fallback for read queries — resets postgrest auth to service role."""
        self.client.postgrest.auth(self._service_key)
        return self.client.table(table)

    @property
    def auth(self):
        """Access Supabase auth methods."""
        return self.client.auth
    
    # ==================== User Operations ====================
    
    async def create_user(self, user_id: str, email: str, full_name: Optional[str] = None) -> None:
        """Upsert a row into public.users so FK constraints from user_credits/jobs are satisfied."""
        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    self._rest_url("users"),
                    headers={**self._admin_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
                    json={"id": user_id, "email": email, "full_name": full_name or ""},
                )
                if not r.is_success:
                    logger.warning(f"create_user DB write skipped (RLS/schema): {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.warning(f"create_user non-fatal error: {e}")

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by ID."""
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    self._rest_url("users"),
                    headers=self._admin_headers(),
                    params={"id": f"eq.{user_id}", "select": "*"}
                )
                if resp.is_success and resp.json():
                    return resp.json()[0]
        except Exception as e:
            logger.warning(f"get_user error: {e}")
        return None
    
    async def create_user_credits(
        self, 
        user_id: str, 
        initial_credits: int = 0
    ) -> Dict[str, Any]:
        """Create initial credits record for a new user."""
        data = {
            "user_id": user_id,
            "credits": initial_credits,
            "credits_used_this_month": 0,
            "subscription_tier": "free"
        }
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    self._rest_url("user_credits"),
                    headers={**self._admin_headers(), "Prefer": "return=representation"},
                    json=data,
                )
                if resp.is_success:
                    return resp.json()[0] if resp.json() else data
                logger.warning(f"create_user_credits DB write skipped (RLS/schema): {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"create_user_credits non-fatal error: {e}")
        return data
    
    async def get_user_credits(self, user_id: str) -> int:
        """Get user's current credit balance. Returns 0 on any error (fail-safe)."""
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    self._rest_url("user_credits"),
                    headers=self._admin_headers(),
                    params={"user_id": f"eq.{user_id}", "select": "credits"}
                )
                if resp.is_success and resp.json():
                    return resp.json()[0]["credits"]
        except Exception as e:
            logger.warning(f"get_user_credits error: {e}")
        return 0
    
    async def get_user_credits_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get full credits data for user."""
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    self._rest_url("user_credits"),
                    headers=self._admin_headers(),
                    params={"user_id": f"eq.{user_id}", "select": "*"}
                )
                if resp.is_success and resp.json():
                    return resp.json()[0]
        except Exception as e:
            logger.warning(f"get_user_credits_data non-fatal error: {e}")
        return None
    
    async def add_user_credits(self, user_id: str, credits: int) -> None:
        """Add credits to user's account."""
        try:
            async with httpx.AsyncClient() as http:
                await http.post(
                    self._rest_url("rpc/add_credits"),
                    headers=self._admin_headers(),
                    json={"p_user_id": user_id, "p_credits": credits}
                )
        except Exception as e:
            logger.error(f"add_user_credits failed: {e}")
    
    async def deduct_user_credits(self, user_id: str, credits: int) -> bool:
        """Deduct credits from user's account. Returns False if insufficient."""
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    self._rest_url("rpc/deduct_credits"),
                    headers=self._admin_headers(),
                    json={"p_user_id": user_id, "p_credits": credits}
                )
                return resp.is_success and resp.text.strip() == "true"
        except Exception:
            return False
    
    async def save_user_voice(self, user_id: str, voice_id: str) -> None:
        """Save cloned voice ID for user."""
        self.client.table("user_voices").upsert({
            "user_id": user_id,
            "voice_id": voice_id,
            "created_at": "now()"
        }).execute()
    
    # ==================== Job Operations ====================
    
    async def create_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new job record."""
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                self._rest_url("jobs"),
                headers={**self._admin_headers(), "Prefer": "return=representation"},
                json=job_data,
            )
            if resp.is_success and resp.json():
                return resp.json()[0]
            logger.warning(f"create_job failed: {resp.status_code} {resp.text[:200]}")
            return job_data
    
    async def get_job(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID, ensuring user ownership."""
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                self._rest_url("jobs"),
                headers=self._admin_headers(),
                params={"job_id": f"eq.{job_id}", "user_id": f"eq.{user_id}", "select": "*"},
            )
            data = resp.json() if resp.is_success else []
            return data[0] if data else None

    async def get_job_public(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID without user ownership check (for public share pages)."""
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                self._rest_url("jobs"),
                headers=self._admin_headers(),
                params={"job_id": f"eq.{job_id}", "select": "*"},
            )
            data = resp.json() if resp.is_success else []
            return data[0] if data else None

    async def get_job_by_runpod_id(self, runpod_id: str) -> Optional[Dict[str, Any]]:
        """Get job by RunPod job ID."""
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                self._rest_url("jobs"),
                headers=self._admin_headers(),
                params={"runpod_id": f"eq.{runpod_id}", "select": "*"},
            )
            data = resp.json() if resp.is_success else []
            return data[0] if data else None
    
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update job status."""
        data: Dict[str, Any] = {"status": status}
        if additional_data:
            data.update(additional_data)
        async with httpx.AsyncClient() as http:
            await http.patch(
                self._rest_url("jobs"),
                headers={**self._admin_headers(), "Prefer": "return=minimal"},
                params={"job_id": f"eq.{job_id}"},
                json=data,
            )

    async def update_job(self, job_id: str, data: Dict[str, Any]) -> None:
        """Update job with arbitrary data."""
        async with httpx.AsyncClient() as http:
            await http.patch(
                self._rest_url("jobs"),
                headers={**self._admin_headers(), "Prefer": "return=minimal"},
                params={"job_id": f"eq.{job_id}"},
                json=data,
            )
            
    async def cleanup_zombie_jobs(self) -> int:
        """Find jobs stuck in 'processing' or 'queued' for > 15 mins and mark as failed."""
        from datetime import datetime, timedelta
        cutoff_time = (datetime.utcnow() - timedelta(minutes=15)).isoformat()
        
        async with httpx.AsyncClient() as http:
            # We must use PostgREST filters to find stale jobs
            params = {
                "status": "in.(queued,processing,pending)",
                "updated_at": f"lt.{cutoff_time}",
                "select": "job_id,user_id,credits_estimated,credits_deducted"
            }
            resp = await http.get(
                self._rest_url("jobs"),
                headers=self._admin_headers(),
                params=params
            )
            if not resp.is_success:
                return 0
                
            zombies = resp.json()
            count = 0
            
            for job in zombies:
                job_id = job.get("job_id")
                if job_id:
                    # Update status to failed
                    await self.update_job_status(job_id, "failed", {"error_message": "Job timed out (zombie cleanup)"})
                    count += 1
                    
            return count
    async def list_user_jobs(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List jobs for user with pagination."""
        offset = (page - 1) * page_size
        params: Dict[str, Any] = {
            "user_id": f"eq.{user_id}",
            "select": "*",
            "order": "created_at.desc",
            "offset": offset,
            "limit": page_size,
        }
        if status_filter:
            params["status"] = f"eq.{status_filter}"
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                self._rest_url("jobs"),
                headers={**self._admin_headers(), "Prefer": "count=exact"},
                params=params,
            )
            data = resp.json() if resp.is_success else []
            total = int(resp.headers.get("content-range", "0/0").split("/")[-1] or 0)
            return data, total
    
    async def list_jobs_by_status(
        self,
        status: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List all jobs with a given status (no user filter). Used by worker."""
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                self._rest_url("jobs"),
                headers=self._admin_headers(),
                params={"status": f"eq.{status}", "select": "*", "order": "created_at.asc", "limit": limit},
            )
            return resp.json() if resp.is_success else []

    async def delete_job(self, job_id: str, user_id: str) -> None:
        """Delete a job record."""
        async with httpx.AsyncClient() as http:
            await http.delete(
                self._rest_url("jobs"),
                headers=self._admin_headers(),
                params={"job_id": f"eq.{job_id}", "user_id": f"eq.{user_id}"},
            )
    
    async def finalize_job_credits(
        self,
        job_id: str,
        user_id: str,
        credits_used: int
    ) -> None:
        """Finalize job and deduct credits (fully async — never blocks event loop)."""
        async with httpx.AsyncClient() as http:
            await http.patch(
                self._rest_url("jobs"),
                headers=self._admin_headers(),
                params={"job_id": f"eq.{job_id}"},
                json={"credits_used": credits_used, "status": "completed"},
            )
        await self.deduct_user_credits(user_id, credits_used)

    async def refund_job_credits(self, job_id: str, user_id: str) -> None:
        """Refund credits for a failed/cancelled job (fully async)."""
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                self._rest_url("jobs"),
                headers=self._admin_headers(),
                params={"job_id": f"eq.{job_id}", "select": "credits_estimated,credits_used"},
            )
        data = resp.json() if resp.is_success else []
        if data:
            credits = data[0].get("credits_used") or data[0].get("credits_estimated", 0)
            if credits > 0:
                await self.add_user_credits(user_id, credits)
    
    # ==================== Transaction Operations ====================
    
    async def get_user_transactions(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get user's transaction history."""
        offset = (page - 1) * page_size
        response = self.client.table("transactions").select("*", count="exact").eq("user_id", user_id).order("created_at", desc=True).range(offset, offset + page_size - 1).execute()
        
        return response.data, response.count or 0
    
    async def create_transaction(
        self,
        user_id: str,
        transaction_type: str,
        credits: int,
        description: str,
        stripe_payment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a transaction record."""
        data = {
            "user_id": user_id,
            "type": transaction_type,
            "credits": credits,
            "description": description,
            "stripe_payment_id": stripe_payment_id
        }
        response = self.client.table("transactions").insert(data).execute()
        return response.data[0] if response.data else data

    # ==================== Admin & Analytics ====================
    
    async def get_admin_stats(self) -> Dict[str, Any]:
        """Get global admin statistics."""
        users_resp = self.client.table("users").select("*", count="exact").limit(1).execute()
        jobs_resp = self.client.table("jobs").select("*", count="exact").limit(1).execute()
        transactions_resp = self.client.table("transactions").select("credits").eq("type", "purchase").execute()
        revenue = sum((t.get("credits", 0) / 100) * 10 for t in transactions_resp.data) if transactions_resp.data else 0
        
        return {
            "totalUsers": users_resp.count or 0,
            "totalJobs": jobs_resp.count or 0,
            "totalRevenue": revenue,
            "activeSubscriptions": 0
        }

    async def get_analytics_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics statistics."""
        jobs_resp = self.client.table("jobs").select("id, status, created_at, credits_used", count="exact").eq("user_id", user_id).execute()
        credits_resp = self.client.table("user_credits").select("credits_used_this_month").eq("user_id", user_id).execute()
        used_credits = credits_resp.data[0]["credits_used_this_month"] if credits_resp.data else 0
        
        return {
            "totalJobs": jobs_resp.count or 0,
            "creditsUsed": used_credits,
            "recentJobs": jobs_resp.data[:10] if jobs_resp.data else []
        }

    # ==================== Marketplace & Subscriptions ====================
    
    async def get_marketplace_items(self) -> List[Dict[str, Any]]:
        """Get all marketplace items."""
        resp = self.client.table("marketplace_items").select("*").order("created_at", desc=True).execute()
        return resp.data or []

    async def purchase_marketplace_item(self, user_id: str, item_id: str, price: int) -> bool:
        """Purchase a marketplace item using credits."""
        success = await self.deduct_user_credits(user_id, price)
        if success:
            try:
                self.client.table("user_purchases").insert({
                    "user_id": user_id,
                    "item_id": item_id,
                    "purchase_price": price
                }).execute()
            except Exception as e:
                # Refund if purchase record fails
                await self.add_user_credits(user_id, price)
                return False
        return success

    async def update_subscription_status(self, user_id: str, status: str, current_period_end: Optional[str] = None, stripe_sub_id: Optional[str] = None) -> None:
        """Update a user's subscription status."""
        update_data = {"subscription_status": status}
        if current_period_end:
            update_data["current_period_end"] = current_period_end
        if stripe_sub_id:
            update_data["stripe_subscription_id"] = stripe_sub_id
            
        self.client.table("user_credits").update(update_data).eq("user_id", user_id).execute()


@lru_cache()
def get_supabase() -> SupabaseClient:
    """Get cached Supabase client instance."""
    return SupabaseClient()
