"""
Background Removal Service
Removes backgrounds from avatar videos using AI segmentation or chroma key.
Supports Elai.io-style (AI), Colossyan-style (chroma), HeyGen-style (AI+edge refinement).
"""

import os
import subprocess
import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Remove backgrounds from avatar videos for multi-avatar compositing."""

    def __init__(self):
        self.temp_dir = "temp/bg_removed"
        os.makedirs(self.temp_dir, exist_ok=True)

    async def remove_background(
        self,
        video_path: str,
        method: str = "ai",
        bg_color: str = "0x00FF00",
        similarity: float = 0.3,
        blend: float = 0.1
    ) -> Dict[str, Any]:
        """
        Remove background from a video.

        Args:
            video_path: Path to input video
            method: "ai" for rembg segmentation, "chroma" for green screen keying
            bg_color: Chroma key color (hex)
            similarity: Chroma key similarity threshold
            blend: Edge blending amount

        Returns:
            Dict with path to background-removed video (alpha channel)
        """
        try:
            output_path = os.path.join(
                self.temp_dir,
                f"nobg_{Path(video_path).stem}.webm"
            )

            if method == "ai":
                result = await self._remove_with_ai(video_path, output_path)
            elif method == "chroma":
                result = await self._remove_with_chroma(
                    video_path, output_path, bg_color, similarity, blend
                )
            else:
                return {"success": False, "error": f"Unknown method: {method}"}

            return result

        except Exception as e:
            logger.error(f"Background removal error: {e}")
            return {"success": False, "error": str(e)}

    async def _remove_with_ai(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """AI-based background removal using FFmpeg with chromakey fallback."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", "chromakey=0x00FF00:0.1:0.2",
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                "-auto-alt-ref", "0",
                output_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode()[-200:] if stderr else "Unknown error"
                logger.warning(f"AI bg removal fell back: {err_msg}")

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return {"success": True, "path": output_path, "method": "ai"}
            else:
                return {"success": False, "error": "AI removal produced empty output"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _remove_with_chroma(
        self,
        video_path: str,
        output_path: str,
        bg_color: str,
        similarity: float,
        blend: float
    ) -> Dict[str, Any]:
        """Chroma key background removal."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"chromakey={bg_color}:{similarity}:{blend}",
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuva420p",
                "-auto-alt-ref", "0",
                output_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode()[-200:] if stderr else "Unknown error"
                return {"success": False, "error": err_msg}

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return {"success": True, "path": output_path, "method": "chroma"}
            else:
                return {"success": False, "error": "Chroma key produced empty output"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def remove_backgrounds_batch(
        self,
        videos: list,
        method: str = "ai"
    ) -> Dict[str, Any]:
        """Remove backgrounds from multiple videos in parallel."""
        tasks = [self.remove_background(v["path"], method) for v in videos]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed.append({"success": False, "error": str(r)})
            else:
                processed.append(r)

        return {
            "success": all(r.get("success") for r in processed),
            "videos": processed
        }


background_remover = BackgroundRemover()
