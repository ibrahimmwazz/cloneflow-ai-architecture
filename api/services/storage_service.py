"""
CloneFlow AI - Async Object Storage Offloading
Uploads generated media to Cloudflare R2 / Azure Blob / DigitalOcean Spaces
and returns public CDN URLs to free up local disk + bandwidth.
"""

import os
import logging
import aiofiles
from typing import Optional, Dict
from urllib.parse import urljoin
from dotenv import load_dotenv
import aioboto3

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class StorageService:
    """
    Async storage uploader.
    Supports Cloudflare R2 (S3-compatible) out of the box.
    Extendable to Azure Blob or DigitalOcean Spaces.
    """

    def __init__(self):
        self.provider = os.getenv("STORAGE_PROVIDER", "r2").lower()
        self.bucket = os.getenv("R2_BUCKET_NAME", "cloneflow-media")
        self.public_url = os.getenv("R2_PUBLIC_URL", "").rstrip("/")

        # S3-compatible credentials (R2, Spaces, Minio)
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "")
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "")
        self.account_id = os.getenv("R2_ACCOUNT_ID", "")
        # R2 endpoint format: https://<account-id>.r2.cloudflarestorage.com
        self.endpoint = f"https://{self.account_id}.r2.cloudflarestorage.com" if self.account_id else ""

        # Async S3 client
        self._session = None

    @property
    async def session(self):
        """Create aioboto3 session for R2."""
        if self._session is None:
            self._session = aioboto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="us-east-1"  # R2 uses us-east-1
            )
        return self._session

    async def upload_file(
        self,
        local_path: str,
        remote_key: str,
        content_type: str = "application/octet-stream"
    ) -> Dict:
        """
        Upload a local file to storage and return public URL.
        Automatically deletes local file after successful upload.
        """
        try:
            # Read file
            async with aiofiles.open(local_path, "rb") as f:
                data = await f.read()

            # Upload via S3
            result = await self.upload_bytes(data, remote_key, content_type)
            
            if result["success"]:
                # Delete local file to save disk
                os.remove(local_path)
            
            return result

        except Exception as e:
            logger.error(f"Storage upload error: {e}")
            return {"success": False, "error": str(e)}

    async def upload_bytes(
        self,
        data: bytes,
        remote_key: str,
        content_type: str = "application/octet-stream"
    ) -> Dict:
        """Upload raw bytes directly (no local file)."""
        try:
            session = await self.session
            
            async with session.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="us-east-1"
            ) as s3:
                await s3.put_object(
                    Bucket=self.bucket,
                    Key=remote_key,
                    Body=data,
                    ContentType=content_type
                )

            public_url = f"{self.public_url}/{remote_key}" if self.public_url else f"{self.endpoint}/{self.bucket}/{remote_key}"
            return {"success": True, "url": public_url, "key": remote_key, "size_bytes": len(data)}

        except Exception as e:
            logger.error(f"Storage upload error: {e}")
            return {"success": False, "error": str(e)}

    async def close(self):
        # aioboto3 session doesn't need explicit cleanup
        self._session = None


# Global instance
storage_service = StorageService()
