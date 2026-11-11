"""
Video Generation Service — Lightning AI GPU for LivePortrait face animation
"""

import os
import logging
from typing import Optional, Dict, Any
from api.services.lightning_ai_client import LightningAIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoService:
    """Video generation — GPU via Lightning AI (LivePortrait)"""

    def __init__(self):
        self.lightning = LightningAIClient()
        self.output_dir = "outputs/videos"
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("VideoService initialized (Lightning AI GPU)")

    async def animate_portrait(
        self,
        image_url: str,
        driving_video_url: str,
        fps: int = 30,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Animate a portrait image using LivePortrait on Lightning AI GPU.

        Args:
            image_url: Source portrait image URL
            driving_video_url: Driving motion video URL
            fps: Output video FPS
            user_id: Optional user ID for tracking

        Returns:
            Dict with generated video URL
        """
        try:
            logger.info(f"Generating LivePortrait video for user={user_id}")

            result = await self.lightning.generate_video(
                image_url=image_url,
                driving_video_url=driving_video_url,
                fps=fps
            )

            if not result.get("success"):
                return result

            return {
                "success": True,
                "video_url": result.get("video_url"),
                "local_path": result.get("local_path"),
                "provider": "lightning_ai",
                "model": "liveportrait",
                "fps": fps,
                "user_id": user_id
            }

        except Exception as e:
            logger.error(f"Video generation error: {e}")
            return {"success": False, "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        return await self.lightning.health_check()


# Global instance
video_service = VideoService()
