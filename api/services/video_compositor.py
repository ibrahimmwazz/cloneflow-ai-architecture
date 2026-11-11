"""
Video Compositor Service
Composites multiple avatar videos into a single scene using FFmpeg.
Supports 3 layout styles:
  - elai: Side-by-side conversational (2 avatars)
  - colossyan: Multi-avatar scenario (up to 4, free positioning)
  - heygen: Avatar interaction with transitions and camera angles
"""

import os
import subprocess
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LayoutStyle(str, Enum):
    ELAI = "elai"           # Side-by-side, 2 avatars, conversational
    COLOSSYAN = "colossyan"  # Up to 4 avatars, scenario-based
    HEYGEN = "heygen"        # Interactive, transitions, camera angles


class Position:
    """Avatar position in the scene."""
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def to_overlay_str(self, input_idx: int) -> str:
        return (
            f"[{input_idx}:v]scale={self.width}:{self.height}:force_original_aspect_ratio=1,"
            f"format=rgba[scaled{input_idx}]"
        )


class VideoCompositor:
    """Composite multiple avatar videos into a single scene."""

    def __init__(self):
        self.temp_dir = "temp/composited"
        os.makedirs(self.temp_dir, exist_ok=True)

    async def composite_scene(
        self,
        avatar_videos: List[Dict[str, Any]],
        background_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        style: LayoutStyle = LayoutStyle.ELAI,
        output_path: Optional[str] = None,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30
    ) -> Dict[str, Any]:
        """
        Composite multiple avatar videos into a single scene.

        Each avatar_video dict:
        {
            "path": str,              # Video file path (with alpha channel)
            "position": {x, y, w, h}, # Position in scene
            "z_index": int            # Layer order (0 = bottom)
        }

        Args:
            avatar_videos: List of avatar video configs
            background_path: Background image or video path
            audio_path: Mixed audio path
            style: Layout style
            output_path: Output file path
            resolution: Output resolution (width, height)
            fps: Output FPS

        Returns:
            Dict with composited video path
        """
        try:
            if not avatar_videos:
                return {"success": False, "error": "No avatar videos provided"}

            if output_path is None:
                style_suffix = style.value
                output_path = os.path.join(
                    self.temp_dir,
                    f"scene_{style_suffix}_{os.urandom(4).hex()}.mp4"
                )

            positions = self._calculate_positions(style, len(avatar_videos), resolution)

            filter_complex = self._build_filter_complex(
                avatar_videos, positions, background_path, resolution, fps
            )

            cmd = ["ffmpeg", "-y"]

            if background_path:
                cmd.extend(["-i", background_path])

            for av in avatar_videos:
                cmd.extend(["-i", av["path"]])

            if audio_path:
                cmd.extend(["-i", audio_path])

            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[outv]"
            ])

            if audio_path:
                bg_offset = 1 if background_path else 0
                av_count = len(avatar_videos)
                audio_input_idx = bg_offset + av_count
                cmd.extend(["-map", f"{audio_input_idx}:a"])
            else:
                cmd.extend(["-an"])

            cmd.extend([
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-r", str(fps),
                output_path
            ])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode()[-300:] if stderr else "Unknown error"
                return {"success": False, "error": err_msg}

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return {
                    "success": True,
                    "path": output_path,
                    "style": style.value,
                    "resolution": resolution,
                    "num_avatars": len(avatar_videos)
                }
            else:
                return {"success": False, "error": "Composited video is empty"}

        except Exception as e:
            logger.error(f"Video compositing error: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_positions(
        self,
        style: LayoutStyle,
        num_avatars: int,
        resolution: Tuple[int, int]
    ) -> List[Position]:
        """Calculate avatar positions based on layout style."""
        w, h = resolution

        if style == LayoutStyle.ELAI:
            return self._elai_positions(num_avatars, w, h)
        elif style == LayoutStyle.COLOSSYAN:
            return self._colossyan_positions(num_avatars, w, h)
        elif style == LayoutStyle.HEYGEN:
            return self._heygen_positions(num_avatars, w, h)
        else:
            return self._elai_positions(num_avatars, w, h)

    def _elai_positions(self, num: int, w: int, h: int) -> List[Position]:
        """Elai.io style: Side-by-side, conversational, 2 avatars."""
        avatar_w = int(w * 0.42)
        avatar_h = int(h * 0.75)
        y = int(h * 0.12)

        if num == 2:
            return [
                Position(x=int(w * 0.04), y=y, width=avatar_w, height=avatar_h),
                Position(x=int(w * 0.54), y=y, width=avatar_w, height=avatar_h),
            ]
        elif num == 3:
            avatar_w = int(w * 0.28)
            return [
                Position(x=int(w * 0.02), y=y, width=avatar_w, height=avatar_h),
                Position(x=int(w * 0.36), y=y, width=avatar_w, height=avatar_h),
                Position(x=int(w * 0.70), y=y, width=avatar_w, height=avatar_h),
            ]
        else:
            return self._colossyan_positions(num, w, h)

    def _colossyan_positions(self, num: int, w: int, h: int) -> List[Position]:
        """Colossyan style: Scenario-based, up to 4 avatars, free positioning."""
        if num == 1:
            return [Position(x=int(w * 0.25), y=int(h * 0.05), width=int(w * 0.5), height=int(h * 0.85))]
        elif num == 2:
            return [
                Position(x=int(w * 0.05), y=int(h * 0.1), width=int(w * 0.42), height=int(h * 0.7)),
                Position(x=int(w * 0.53), y=int(h * 0.1), width=int(w * 0.42), height=int(h * 0.7)),
            ]
        elif num == 3:
            return [
                Position(x=int(w * 0.02), y=int(h * 0.15), width=int(w * 0.30), height=int(h * 0.6)),
                Position(x=int(w * 0.35), y=int(h * 0.05), width=int(w * 0.30), height=int(h * 0.7)),
                Position(x=int(w * 0.68), y=int(h * 0.15), width=int(w * 0.30), height=int(h * 0.6)),
            ]
        elif num >= 4:
            return [
                Position(x=int(w * 0.02), y=int(h * 0.05), width=int(w * 0.22), height=int(h * 0.42)),
                Position(x=int(w * 0.26), y=int(h * 0.05), width=int(w * 0.22), height=int(h * 0.42)),
                Position(x=int(w * 0.52), y=int(h * 0.05), width=int(w * 0.22), height=int(h * 0.42)),
                Position(x=int(w * 0.76), y=int(h * 0.05), width=int(w * 0.22), height=int(h * 0.42)),
            ][:num]
        return []

    def _heygen_positions(self, num: int, w: int, h: int) -> List[Position]:
        """HeyGen style: Interactive, dynamic positioning with focus on active speaker."""
        if num == 2:
            return [
                Position(x=int(w * 0.02), y=int(h * 0.08), width=int(w * 0.46), height=int(h * 0.78)),
                Position(x=int(w * 0.52), y=int(h * 0.08), width=int(w * 0.46), height=int(h * 0.78)),
            ]
        elif num == 3:
            return [
                Position(x=int(w * 0.01), y=int(h * 0.12), width=int(w * 0.32), height=int(h * 0.65)),
                Position(x=int(w * 0.34), y=int(h * 0.04), width=int(w * 0.32), height=int(h * 0.78)),
                Position(x=int(w * 0.67), y=int(h * 0.12), width=int(w * 0.32), height=int(h * 0.65)),
            ]
        elif num >= 4:
            return [
                Position(x=int(w * 0.01), y=int(h * 0.08), width=int(w * 0.23), height=int(h * 0.40)),
                Position(x=int(w * 0.26), y=int(h * 0.08), width=int(w * 0.23), height=int(h * 0.40)),
                Position(x=int(w * 0.51), y=int(h * 0.08), width=int(w * 0.23), height=int(h * 0.40)),
                Position(x=int(w * 0.76), y=int(h * 0.08), width=int(w * 0.23), height=int(h * 0.40)),
            ][:num]
        return []

    def _build_filter_complex(
        self,
        avatar_videos: List[Dict[str, Any]],
        positions: List[Position],
        background_path: Optional[str],
        resolution: Tuple[int, int],
        fps: int
    ) -> str:
        """Build FFmpeg filter_complex string for scene compositing."""
        w, h = resolution
        parts = []

        bg_idx = 0 if background_path else -1
        av_start = 1 if background_path else 0

        for i, (av, pos) in enumerate(zip(avatar_videos, positions)):
            input_idx = av_start + i
            parts.append(
                f"[{input_idx}:v]scale={pos.width}:{pos.height}:"
                f"force_original_aspect_ratio=1,format=rgba,"
                f"fps={fps}[scaled{i}]"
            )

        if background_path:
            parts.append(f"[0:v]scale={w}:{h},fps={fps}[bg]")
            current = "[bg]"
        else:
            parts.append(f"color=black:{w}x{h}:{fps}[bg]")
            current = "[bg]"

        for i in range(len(avatar_videos)):
            pos = positions[i]
            next_tmp = f"[tmp{i}]"
            parts.append(
                f"{current}[scaled{i}]overlay={pos.x}:{pos.y}:format=auto{next_tmp}"
            )
            current = next_tmp

        parts.append(f"{current}format=yuv420p[outv]")

        return ";".join(parts)


video_compositor = VideoCompositor()
