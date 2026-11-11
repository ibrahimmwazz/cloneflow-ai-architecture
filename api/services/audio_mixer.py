"""
Audio Mixer Service
Mixes multiple audio tracks with timing, volume control, and crossfade.
Supports conversational turn-based dialogue for multi-avatar scenes.
"""

import os
import subprocess
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioMixer:
    """Mix multiple audio tracks for multi-avatar conversations."""

    def __init__(self):
        self.temp_dir = "temp/audio_mix"
        os.makedirs(self.temp_dir, exist_ok=True)

    async def mix_conversation(
        self,
        tracks: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        crossfade_ms: int = 200,
        silence_pad_ms: int = 300
    ) -> Dict[str, Any]:
        """
        Mix multiple audio tracks into a single conversation audio.

        Each track dict:
        {
            "path": str,           # Audio file path
            "start_ms": int,       # Start time in milliseconds
            "duration_ms": int,    # Duration in milliseconds
            "volume": float,       # Volume multiplier (0.0-2.0, default 1.0)
            "avatar_id": str       # Which avatar is speaking
        }

        Args:
            tracks: List of audio track configs
            output_path: Output file path (auto-generated if None)
            crossfade_ms: Crossfade duration between tracks
            silence_pad_ms: Silence padding between tracks

        Returns:
            Dict with mixed audio path and metadata
        """
        try:
            if not tracks:
                return {"success": False, "error": "No tracks provided"}

            if output_path is None:
                output_path = os.path.join(self.temp_dir, "mixed_conversation.wav")

            sorted_tracks = sorted(tracks, key=lambda t: t.get("start_ms", 0))

            total_duration = max(
                t.get("start_ms", 0) + t.get("duration_ms", 0)
                for t in sorted_tracks
            )

            filter_parts = []
            for i, track in enumerate(sorted_tracks):
                path = track["path"]
                start_sec = track.get("start_ms", 0) / 1000.0
                duration_sec = track.get("duration_ms", 0) / 1000.0
                volume = track.get("volume", 1.0)

                delay_ms = int(start_sec * 1000)
                fade_in = min(crossfade_ms, int(duration_sec * 1000) // 4)
                fade_out = min(crossfade_ms, int(duration_sec * 1000) // 4)

                filter_parts.append(
                    f"[{i}:a]adelay={delay_ms}|{delay_ms},"
                    f"afade=t=in:d={fade_in / 1000.0:.3f},"
                    f"afade=t=out:st={(duration_sec - fade_out / 1000.0):.3f}:d={fade_out / 1000.0:.3f},"
                    f"volume={volume}[a{i}]"
                )

            mix_inputs = " ".join(f"[a{i}]" for i in range(len(sorted_tracks)))
            amix_inputs = len(sorted_tracks)
            filter_parts.append(f"{mix_inputs}amix=inputs={amix_inputs}:duration=longest[out]")

            filter_complex = ";".join(filter_parts)

            cmd = ["ffmpeg", "-y"]
            for track in sorted_tracks:
                cmd.extend(["-i", track["path"]])
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-ac", "2",
                "-ar", "44100",
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
                    "duration_ms": total_duration,
                    "num_tracks": len(tracks)
                }
            else:
                return {"success": False, "error": "Mixed audio is empty"}

        except Exception as e:
            logger.error(f"Audio mixing error: {e}")
            return {"success": False, "error": str(e)}

    async def add_background_music(
        self,
        speech_path: str,
        music_path: str,
        music_volume: float = 0.15,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add background music to a speech track."""
        try:
            if output_path is None:
                output_path = os.path.join(
                    self.temp_dir,
                    f"with_music_{Path(speech_path).stem}.wav"
                )

            cmd = [
                "ffmpeg", "-y",
                "-i", speech_path,
                "-i", music_path,
                "-filter_complex",
                f"[1:a]volume={music_volume}[bg];[0:a][bg]amix=inputs=2:duration=first",
                "-ac", "2",
                "-ar", "44100",
                output_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return {"success": True, "path": output_path}
            else:
                return {"success": False, "error": "Music mix produced empty output"}

        except Exception as e:
            logger.error(f"Background music error: {e}")
            return {"success": False, "error": str(e)}


audio_mixer = AudioMixer()
