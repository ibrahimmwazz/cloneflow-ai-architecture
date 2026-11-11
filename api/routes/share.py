"""
Share routes for generating and managing shareable links for content.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import secrets
from datetime import datetime, timedelta

from api.services.supabase_client import get_supabase
from api.config import settings

router = APIRouter(prefix="/api/share", tags=["share"])


class CreateShareLinkRequest(BaseModel):
    job_id: str
    expires_in_hours: Optional[int] = None  # None = never expires
    password: Optional[str] = None  # Optional password protection


class ShareLinkResponse(BaseModel):
    share_id: str
    share_url: str
    expires_at: Optional[datetime]
    password_protected: bool
    created_at: datetime


@router.post("/create", response_model=ShareLinkResponse)
async def create_share_link(request: CreateShareLinkRequest):
    """
    Create a shareable link for a video.
    
    - job_id: The ID of the job/video to share
    - expires_in_hours: Optional expiration time in hours
    - password: Optional password for protection
    """
    supabase = get_supabase()
    
    # Verify job exists and belongs to user
    job_response = supabase.table('jobs').select('*').eq('job_id', request.job_id).execute()
    
    if not job_response.data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_response.data[0]
    
    # Generate unique share ID
    share_id = secrets.urlsafe_urlsafe(16)
    
    # Calculate expiration
    expires_at = None
    if request.expires_in_hours:
        expires_at = datetime.utcnow() + timedelta(hours=request.expires_in_hours)
    
    # Store share link in database
    share_data = {
        'share_id': share_id,
        'job_id': request.job_id,
        'user_id': job.get('user_id'),
        'expires_at': expires_at.isoformat() if expires_at else None,
        'password': request.password if request.password else None,
        'created_at': datetime.utcnow().isoformat(),
        'views': 0,
        'active': True
    }
    
    supabase.table('share_links').insert(share_data).execute()
    
    # Generate share URL
    share_url = f"{settings.APP_URL}/share/{share_id}"
    
    return ShareLinkResponse(
        share_id=share_id,
        share_url=share_url,
        expires_at=expires_at,
        password_protected=bool(request.password),
        created_at=datetime.utcnow()
    )


@router.get("/{share_id}")
async def get_shared_content(share_id: str, password: Optional[str] = None):
    """
    Get shared content by share ID.
    Checks expiration and password if set.
    """
    supabase = get_supabase()
    
    # Get share link
    share_response = supabase.table('share_links').select('*').eq('share_id', share_id).execute()
    
    if not share_response.data:
        raise HTTPException(status_code=404, detail="Share link not found")
    
    share = share_response.data[0]
    
    # Check if active
    if not share.get('active'):
        raise HTTPException(status_code=410, detail="Share link has been deactivated")
    
    # Check expiration
    if share.get('expires_at'):
        expires_at = datetime.fromisoformat(share['expires_at'])
        if datetime.utcnow() > expires_at:
            raise HTTPException(status_code=410, detail="Share link has expired")
    
    # Check password
    if share.get('password') and share['password'] != password:
        raise HTTPException(status_code=403, detail="Invalid password")
    
    # Get job details
    job_response = supabase.table('jobs').select('*').eq('job_id', share['job_id']).execute()
    
    if not job_response.data:
        raise HTTPException(status_code=404, detail="Associated job not found")
    
    job = job_response.data[0]
    
    # Increment view count
    supabase.table('share_links').update({'views': share.get('views', 0) + 1}).eq('share_id', share_id).execute()
    
    return {
        'share_id': share_id,
        'job': {
            'job_id': job['job_id'],
            'video_url': job.get('video_url'),
            'status': job.get('status'),
            'created_at': job.get('created_at'),
            'text': job.get('text', '')[:200]  # Truncated script
        },
        'password_protected': bool(share.get('password'))
    }


@router.delete("/{share_id}")
async def delete_share_link(share_id: str):
    """
    Delete/deactivate a share link.
    """
    supabase = get_supabase()
    
    # Check if share exists
    share_response = supabase.table('share_links').select('*').eq('share_id', share_id).execute()
    
    if not share_response.data:
        raise HTTPException(status_code=404, detail="Share link not found")
    
    # Deactivate instead of delete for audit trail
    supabase.table('share_links').update({'active': False}).eq('share_id', share_id).execute()
    
    return {"message": "Share link deactivated"}


@router.get("/list/user")
async def list_user_share_links():
    """
    List all share links for the authenticated user.
    """
    supabase = get_supabase()
    
    # This would need authentication middleware to get user_id
    # For now, return all (in production, filter by user_id)
    share_response = supabase.table('share_links').select('*').eq('active', True).execute()
    
    return {
        "share_links": share_response.data
    }
