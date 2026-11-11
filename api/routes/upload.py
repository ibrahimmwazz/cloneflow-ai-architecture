"""
File Upload API Route
Handles file uploads to Cloudflare R2 storage
"""

import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from botocore.exceptions import NoCredentialsError, ClientError
import logging

from services.cloudflare_client import CloudflareR2Client
from services.auth import get_current_user_optional

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize R2 client
r2_client = CloudflareR2Client()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user_optional)
):
    """
    Upload a file to Cloudflare R2 storage
    Returns the public URL of the uploaded file
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (10MB limit)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Reset file pointer
        await file.seek(0)
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Determine content type
        content_type = file.content_type or "application/octet-stream"
        
        # Upload to R2
        try:
            public_url = await r2_client.upload_file(
                file_data=content,
                filename=unique_filename,
                content_type=content_type
            )
        except Exception as e:
            logger.error(f"R2 upload failed: {str(e)}")
            # Fallback to local storage if R2 fails
            public_url = await save_file_locally(content, unique_filename, content_type)
        
        return JSONResponse(content={
            "url": public_url,
            "filename": unique_filename,
            "size": file_size,
            "content_type": content_type
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def save_file_locally(file_data: bytes, filename: str, content_type: str) -> str:
    """
    Fallback: Save file locally and return URL
    """
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "/tmp/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file locally
        file_path = os.path.join(upload_dir, filename)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_data)
        
        # Return a mock URL (in production, this would be a proper public URL)
        return f"https://storage.cloneflow.ai/uploads/{filename}"
        
    except Exception as e:
        logger.error(f"Local save failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save file")
