"""
Multi-Avatar Service
Orchestrates multi-avatar video generation across all 3 styles:
  - Elai.io style: 2+ avatars side-by-side, conversational
  - Colossyan style: Multi-avatar scenarios, up to 4 avatars
  - HeyGen style: Avatar interactions with dynamic focus

Pipeline:
  1. Generate TTS audio for each dialogue turn (parallel)
  2. Generate individual avatar videos (parallel via GPU queue)
  3. Remove backgrounds from each video
  4. Mix audio tracks with timing
  5. Composite final scene
  6. Upload to R2 storage
"""

import os
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from api.services.tts_service import get_tts_service
from api.services.video_service import video_service
from api.services.background_remover import background_remover
from api.services.audio_mixer import audio_mixer
from api.services.video_compositor import video_compositor, LayoutStyle
from api.services.storage_service import storage_service
from api.services.redis_queue import get_redis_queue

logger = logging.getLogger(__name__)


class MultiAvatarService:
    """Orchestrate multi-avatar video generation."""

    def __init__(self):
        self.siliconflow = None  # SiliconFlowClient module was removed
        self.output_dir = "outputs/multi_avatar"
        os.makedirs(self.output_dir, exist_ok=True)

    async def create_conversation(
        self,
        avatars: List[Dict[str, Any]],
        dialogue: List[Dict[str, Any]],
        style: str = "elai",
        background_url: Optional[str] = None,
        background_music_url: Optional[str] = None,
        resolution: str = "1920x1080",
        user_id: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a multi-avatar conversation video.

        Args:
            avatars: List of avatar configs with image_url, name, voice_id
            dialogue: List of dialogue turns with avatar_id, text, emotion
            style: "elai", "colossyan", or "heygen"
            background_url: Optional background image/video URL
            background_music_url: Optional background music URL
            resolution: Output resolution "WxH"
            user_id: User ID for tracking
            job_id: Optional job ID for queue tracking

        Returns:
            Dict with final video URL and metadata
        """
        try:
            if job_id is None:
                job_id = f"multi_{uuid.uuid4().hex[:12]}"

            w, h = map(int, resolution.split("x"))
            layout = LayoutStyle(style)

            if job_id:
                await self._update_job_status(job_id, "processing", 5)

            logger.info(f"Starting multi-avatar conversation: {len(avatars)} avatars, "
                        f"{len(dialogue)} turns, style={style}")

            # Step 1: Generate TTS audio for each dialogue turn (parallel)
            if job_id:
                await self._update_job_status(job_id, "generating_audio", 15)

            audio_tracks = await self._generate_dialogue_audio(dialogue, avatars)

            # Step 2: Calculate timing for the conversation
            if job_id:
                await self._update_job_status(job_id, "calculating_timing", 25)

            timing = self._calculate_timing(audio_tracks)

            # Step 3: Generate individual avatar videos (parallel via GPU queue)
            if job_id:
                await self._update_job_status(job_id, "generating_videos", 35)

            avatar_videos = await self._generate_avatar_videos(
                avatars, audio_tracks, timing
            )

            # Step 4: Remove backgrounds
            if job_id:
                await self._update_job_status(job_id, "removing_backgrounds", 55)

            bg_removed = await self._remove_backgrounds(avatar_videos)

            # Step 5: Mix audio
            if job_id:
                await self._update_job_status(job_id, "mixing_audio", 70)

            mixed_audio = await self._mix_audio(audio_tracks, background_music_url)

            # Step 6: Composite final scene
            if job_id:
                await self._update_job_status(job_id, "compositing_scene", 80)

            composited = await self._composite_scene(
                bg_removed, mixed_audio, background_url, layout, (w, h)
            )

            # Step 7: Upload to R2
            if job_id:
                await self._update_job_status(job_id, "uploading", 90)

            final_url = await self._upload_to_storage(composited, job_id)

            if job_id:
                await self._update_job_status(job_id, "completed", 100)

            return {
                "success": True,
                "video_url": final_url,
                "job_id": job_id,
                "style": style,
                "num_avatars": len(avatars),
                "num_turns": len(dialogue),
                "duration_ms": timing.get("total_duration_ms", 0),
                "created_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Multi-avatar conversation error: {e}")
            if job_id:
                await self._update_job_status(job_id, "failed", 0, str(e))
            return {"success": False, "error": str(e), "job_id": job_id}

    async def _generate_dialogue_audio(
        self,
        dialogue: List[Dict[str, Any]],
        avatars: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate TTS audio for each dialogue turn in parallel."""
        avatar_map = {a.get("id"): a for a in avatars}

        async def generate_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
            avatar = avatar_map.get(turn.get("avatar_id"), avatars[0])
            voice_id = avatar.get("voice_id", "default")

            tts_service = get_tts_service()
            if not tts_service:
                return {
                    "avatar_id": turn["avatar_id"],
                    "text": turn["text"],
                    "path": None,
                    "duration_ms": 2000,
                    "error": "TTS service not available"
                }

            result = await tts_service.synthesize(
                text=turn["text"],
                voice_id=voice_id
            )

            return {
                "avatar_id": turn["avatar_id"],
                "text": turn["text"],
                "emotion": turn.get("emotion", "neutral"),
                "path": result.get("audio_path") if result.get("success") else None,
                "duration_ms": result.get("duration_ms", 2000),
                "error": result.get("error") if not result.get("success") else None
            }

        tasks = [generate_turn(turn) for turn in dialogue]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tracks = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                tracks.append({
                    "avatar_id": dialogue[i].get("avatar_id"),
                    "text": dialogue[i].get("text", ""),
                    "path": None,
                    "duration_ms": 2000,
                    "error": str(r)
                })
            else:
                tracks.append(r)

        return tracks

    def _calculate_timing(self, audio_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate timing for the conversation with pauses between turns."""
        current_ms = 0
        pause_between_turns = 500
        timed_tracks = []

        for track in audio_tracks:
            duration = track.get("duration_ms", 2000)
            timed_tracks.append({
                **track,
                "start_ms": current_ms,
                "end_ms": current_ms + duration
            })
            current_ms += duration + pause_between_turns

        return {
            "tracks": timed_tracks,
            "total_duration_ms": current_ms - pause_between_turns
        }

    async def _generate_avatar_videos(
        self,
        avatars: List[Dict[str, Any]],
        audio_tracks: List[Dict[str, Any]],
        timing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate individual avatar videos for each speaking turn."""
        avatar_turns = {}
        for track in timing["tracks"]:
            aid = track["avatar_id"]
            if aid not in avatar_turns:
                avatar_turns[aid] = []
            avatar_turns[aid].append(track)

        async def generate_avatar_clip(avatar_id: str, turns: List[Dict]) -> Dict[str, Any]:
            avatar = next((a for a in avatars if a.get("id") == avatar_id), avatars[0])

            result = await video_service.animate_portrait(
                image_url=avatar.get("image_url", ""),
                driving_video_url=avatar.get("driving_video_url", ""),
                fps=30,
                user_id=avatar.get("user_id")
            )

            return {
                "avatar_id": avatar_id,
                "name": avatar.get("name", "Avatar"),
                "video_path": result.get("local_path") if result.get("success") else None,
                "video_url": result.get("video_url") if result.get("success") else None,
                "turns": turns,
                "success": result.get("success", False),
                "error": result.get("error") if not result.get("success") else None
            }

        tasks = [generate_avatar_clip(aid, turns) for aid, turns in avatar_turns.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        videos = []
        for r in results:
            if isinstance(r, Exception):
                videos.append({"success": False, "error": str(r)})
            else:
                videos.append(r)

        return videos

    async def _remove_backgrounds(
        self,
        avatar_videos: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove backgrounds from all avatar videos."""
        valid_videos = [v for v in avatar_videos if v.get("success") and v.get("video_path")]

        tasks = [
            background_remover.remove_background(v["video_path"], method="ai")
            for v in valid_videos
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        bg_removed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                bg_removed.append({"success": False, "error": str(r)})
            elif r.get("success"):
                bg_removed.append({
                    **valid_videos[i],
                    "bg_removed_path": r.get("path")
                })
            else:
                bg_removed.append({**valid_videos[i], "bg_removed_path": valid_videos[i]["video_path"]})

        return bg_removed

    async def _mix_audio(
        self,
        audio_tracks: List[Dict[str, Any]],
        background_music_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mix all audio tracks with timing."""
        mix_tracks = [
            {
                "path": t["path"],
                "start_ms": t.get("start_ms", 0),
                "duration_ms": t.get("duration_ms", 2000),
                "volume": 1.0,
                "avatar_id": t["avatar_id"]
            }
            for t in audio_tracks
            if t.get("path")
        ]

        result = await audio_mixer.mix_conversation(mix_tracks)

        if result.get("success") and background_music_url:
            result = await audio_mixer.add_background_music(
                result["path"], background_music_url
            )

        return result

    async def _composite_scene(
        self,
        bg_removed: List[Dict[str, Any]],
        mixed_audio: Dict[str, Any],
        background_url: Optional[str],
        layout: LayoutStyle,
        resolution: tuple
    ) -> Dict[str, Any]:
        """Composite the final multi-avatar scene."""
        avatar_inputs = [
            {
                "path": v.get("bg_removed_path", v.get("video_path", "")),
                "position": None,
                "z_index": i
            }
            for i, v in enumerate(bg_removed)
        ]

        return await video_compositor.composite_scene(
            avatar_videos=avatar_inputs,
            background_path=background_url,
            audio_path=mixed_audio.get("path") if mixed_audio.get("success") else None,
            style=layout,
            resolution=resolution
        )

    async def _upload_to_storage(
        self,
        composited: Dict[str, Any],
        job_id: str
    ) -> Optional[str]:
        """Upload final video to R2 storage."""
        if not composited.get("success") or not composited.get("path"):
            return None

        result = await storage_service.upload_file(
            local_path=composited["path"],
            remote_key=f"multi_avatar/{job_id}/final.mp4",
            content_type="video/mp4"
        )

        return result.get("url") if result.get("success") else None

    async def _update_job_status(
        self,
        job_id: str,
        status: str,
        progress: int,
        error: Optional[str] = None
    ):
        """Update job status in Redis queue."""
        try:
            queue = await get_redis_queue()
            await queue.update_job_status(job_id, {
                "status": status,
                "progress": progress,
                "error": error,
                "updated_at": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to update job status: {e}")


# Lazy instantiation - only create when needed to avoid startup failures
multi_avatar_service = None

def get_multi_avatar_service():
    """Get multi-avatar service instance (lazy initialization)."""
    global multi_avatar_service
    if multi_avatar_service is None:
        try:
            multi_avatar_service = MultiAvatarService()
        except Exception as e:
            logger.warning(f"Failed to initialize MultiAvatarService: {e}")
            return None
    return multi_avatar_service
