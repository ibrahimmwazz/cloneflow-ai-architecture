"""
File Upload API Route
Uploads directly to Cloudflare R2 and returns a real 7-day presigned URL
that can be accessed by SadTalker and any external service.
"""

import os
import uuid
import boto3
from botocore.client import Config
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from api.dependencies.auth import require_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def _r2():
    account = os.getenv("R2_ACCOUNT_ID", "")
    key_id  = os.getenv("R2_ACCESS_KEY_ID", "")
    secret  = os.getenv("R2_SECRET_ACCESS_KEY", "")
    if not (account and key_id and secret):
        raise RuntimeError("R2 credentials not configured")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Depends(require_user),
):
    """
    Upload portrait/audio to Cloudflare R2.
    Returns a 7-day presigned URL — real, publicly accessible by the SadTalker worker.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        content = await file.read()
        file_size = len(content)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
        key = f"uploads/{uuid.uuid4()}{ext}"
        bucket = os.getenv("R2_BUCKET_NAME", "aiclone")

        r2 = _r2()
        r2.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
            ContentType=file.content_type or "application/octet-stream",
        )
        url = r2.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=604800,  # 7 days
        )
        logger.info(f"Uploaded {file.filename} -> R2 key={key} size={file_size}")

        return JSONResponse(content={
            "url": url,
            "filename": file.filename,
            "size": file_size,
            "content_type": file.content_type or "application/octet-stream",
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
