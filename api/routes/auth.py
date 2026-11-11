"""
CloneFlow AI - Authentication Router
Handles user authentication via Supabase GoTrue
"""

from fastapi import APIRouter, HTTPException, Depends, Header, Query
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

from api.services.supabase_client import get_supabase, SupabaseClient

router = APIRouter()


class UserRegister(BaseModel):
    """Registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str


class ForgotPasswordRequest(BaseModel):
    """Forgot password request model."""
    email: EmailStr
    redirect_to: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    """Reset password request model."""
    new_password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User data response."""
    id: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    credits: int = 0


class AuthResponse(BaseModel):
    """Authentication response with tokens."""
    user: UserResponse
    access_token: str
    refresh_token: str
    expires_in: int


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


@router.post("/register", response_model=AuthResponse)
async def register(
    request: UserRegister,
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Register a new user account.
    
    Creates user in Supabase Auth and initializes credits.
    """
    try:
        # sign_up works directly now that email confirmation is disabled in Supabase
        auth_response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {"data": {"full_name": request.full_name or ""}}
        })
        
        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Registration failed")
        
        user_id = auth_response.user.id
        
        # Upsert into public.users first (FK dependency for user_credits and jobs)
        await supabase.create_user(user_id, request.email, request.full_name)
        
        # Initialize user credits (give new users 10 free credits)
        await supabase.create_user_credits(user_id, initial_credits=10)
        
        # With email confirmation off, session is returned immediately
        session = auth_response.session
        if not session:
            # Fallback: sign in to get a session
            session = supabase.auth.sign_in_with_password({
                "email": request.email,
                "password": request.password
            }).session
        
        return AuthResponse(
            user=UserResponse(
                id=user_id,
                email=request.email,
                full_name=request.full_name,
                created_at=datetime.utcnow(),
                credits=10
            ),
            access_token=session.access_token,
            refresh_token=session.refresh_token,
            expires_in=session.expires_in
        )
    
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ("already registered", "already exists", "user already", "duplicate")):
            raise HTTPException(status_code=409, detail="Email already registered")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(
    request: UserLogin,
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Authenticate user and return tokens.
    
    Uses Supabase GoTrue for authentication.
    """
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if not auth_response.user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        
        # Get user credits
        credits_data = await supabase.get_user_credits_data(auth_response.user.id)
        
        return AuthResponse(
            user=UserResponse(
                id=auth_response.user.id,
                email=request.email,
                full_name=auth_response.user.user_metadata.get("full_name"),
                created_at=auth_response.user.created_at if isinstance(auth_response.user.created_at, datetime) else datetime.fromisoformat(auth_response.user.created_at.replace("Z", "+00:00")),
                credits=credits_data.get("credits", 0) if credits_data else 0
            ),
            access_token=auth_response.session.access_token,
            refresh_token=auth_response.session.refresh_token,
            expires_in=auth_response.session.expires_in
        )
    
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        status = 401 if any(x in msg.lower() for x in ("invalid", "credential", "password", "not found", "user")) else 400
        raise HTTPException(status_code=status, detail=msg)


@router.post("/logout")
async def logout(
    authorization: Optional[str] = Header(None),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Logout user and invalidate their specific JWT via Supabase admin API.
    Always returns success to avoid leaking whether a token was valid.
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        try:
            supabase.client.auth.admin.sign_out(token)
        except Exception:
            pass
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(
    request: TokenRefresh,
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Refresh access token using refresh token.
    
    Returns new access and refresh tokens.
    """
    try:
        auth_response = supabase.auth.refresh_session(request.refresh_token)
        
        if not auth_response.user:
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token"
            )
        
        credits_data = await supabase.get_user_credits_data(auth_response.user.id)
        
        return AuthResponse(
            user=UserResponse(
                id=auth_response.user.id,
                email=auth_response.user.email,
                full_name=auth_response.user.user_metadata.get("full_name"),
                created_at=auth_response.user.created_at if isinstance(auth_response.user.created_at, datetime) else datetime.fromisoformat(auth_response.user.created_at.replace("Z", "+00:00")),
                credits=credits_data.get("credits", 0) if credits_data else 0
            ),
            access_token=auth_response.session.access_token,
            refresh_token=auth_response.session.refresh_token,
            expires_in=auth_response.session.expires_in
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    authorization: Optional[str] = Header(None),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Get the currently authenticated user.
    
    Requires Bearer token in Authorization header.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )
    
    token = authorization[7:]
    
    try:
        result = supabase.auth.get_user(token)
        user = result.user
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
        
        credits_data = await supabase.get_user_credits_data(user.id)
        
        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.user_metadata.get("full_name"),
            created_at=user.created_at if isinstance(user.created_at, datetime) else datetime.fromisoformat(user.created_at.replace("Z", "+00:00")),
            credits=credits_data.get("credits", 0) if credits_data else 0
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Send password reset email with optional redirect URL.
    """
    try:
        opts = {}
        if request.redirect_to:
            opts["redirect_to"] = request.redirect_to
        supabase.auth.reset_password_for_email(request.email, opts)
    except Exception:
        pass
    # Always return success to avoid email enumeration
    return {"message": "If that email is registered, a reset link has been sent"}


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    authorization: Optional[str] = Header(None),
    supabase: SupabaseClient = Depends(get_supabase)
):
    """
    Reset password using the recovery token from the email link.
    Uses admin.update_user_by_id so the correct user's password is changed.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Recovery token required")
    recovery_token = authorization[7:]
    try:
        user_resp = supabase.auth.get_user(recovery_token)
        if not user_resp or not user_resp.user:
            raise HTTPException(status_code=401, detail="Invalid or expired recovery token")
        supabase.client.auth.admin.update_user_by_id(
            user_resp.user.id,
            {"password": request.new_password}
        )
        return {"message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
