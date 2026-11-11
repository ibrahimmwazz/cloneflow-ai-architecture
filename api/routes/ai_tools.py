"""
CloneFlow AI - AI Tools Routes
Endpoints for: AI Script Writer, TTS Preview, Background Removal
"""

import asyncio
import os
import io
import uuid
import base64
import logging
import httpx
import boto3
from botocore.client import Config
from fastapi import APIRouter, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

import tempfile
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai-tools"])

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1"

def _r2():
    account = os.getenv("R2_ACCOUNT_ID", "")
    key_id  = os.getenv("R2_ACCESS_KEY_ID", "")
    secret  = os.getenv("R2_SECRET_ACCESS_KEY", "")
    if not (account and key_id and secret):
        raise RuntimeError("R2 credentials not configured (R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def _upload_r2(data: bytes, ext: str, content_type: str) -> str:
    key = f"previews/{uuid.uuid4()}{ext}"
    bucket = os.getenv("R2_BUCKET_NAME", "aiclone")
    r2 = _r2()
    r2.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    return r2.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600
    )


# ── AI Script Writer ──────────────────────────────────────────────────────────

class ScriptRequest(BaseModel):
    topic: str
    tone: str = "professional"        # professional, casual, energetic, educational
    duration_seconds: int = 30        # target video length → word count
    language: str = "en"

class ScriptResponse(BaseModel):
    script: str
    word_count: int
    estimated_duration: int

TONE_PROMPTS = {
    "professional": "formal, clear, confident, business-appropriate",
    "casual": "friendly, conversational, relaxed, approachable",
    "energetic": "enthusiastic, upbeat, exciting, motivating",
    "educational": "informative, clear, structured, easy to understand",
}

SCRIPT_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

def _template_script(topic: str, tone: str, target_words: int) -> str:
    """Zero-dependency fallback script generator using sentence templates."""
    intros = {
        "professional": f"Today, I want to talk about {topic}.",
        "casual": f"Hey everyone! Let's talk about {topic}.",
        "energetic": f"Are you ready? We're diving into {topic} right now!",
        "educational": f"In this video, we'll explore the topic of {topic}.",
    }
    bodies = {
        "professional": (
            f"Understanding {topic} is essential in today's world. "
            f"There are many important aspects to consider when it comes to {topic}. "
            f"Experts have identified key strategies that make a real difference. "
            f"By focusing on the fundamentals, we can achieve outstanding results. "
            f"The latest insights around {topic} show tremendous potential for growth."
        ),
        "casual": (
            f"So {topic} is actually really interesting once you get into it. "
            f"I've been looking into this and there's so much cool stuff to discover. "
            f"A lot of people don't realize how much {topic} can impact everyday life. "
            f"Once you start learning about {topic}, it's hard to stop."
        ),
        "energetic": (
            f"Let me tell you, {topic} is absolutely game-changing! "
            f"This is the kind of thing that transforms how we think and act. "
            f"The energy around {topic} right now is incredible! "
            f"People all over the world are waking up to the power of {topic}!"
        ),
        "educational": (
            f"First, let's define what {topic} actually means. "
            f"{topic} encompasses a wide range of concepts and applications. "
            f"Research has shown that understanding {topic} leads to better outcomes. "
            f"There are three key principles that form the foundation of {topic}. "
            f"By mastering these fundamentals, you'll have a strong grasp of {topic}."
        ),
    }
    outros = {
        "professional": f"In conclusion, {topic} represents a significant opportunity for all of us.",
        "casual": f"Anyway, that's my take on {topic}. Hope you found it useful!",
        "energetic": f"So get out there and make {topic} work for you — you've got this!",
        "educational": f"To summarize, {topic} is a fascinating subject well worth your attention.",
    }
    t = tone if tone in intros else "professional"
    base = f"{intros[t]} {bodies[t]} {outros[t]}"
    words = base.split()
    if len(words) < target_words:
        filler = f"There is so much more to learn about {topic} and its many applications. "
        while len(words) < target_words:
            words.extend(filler.split())
    return " ".join(words[:target_words])

@router.post("/script", response_model=ScriptResponse)
@limiter.limit("20/minute")
async def generate_script(req: ScriptRequest, request: Request):
    """Generate a spoken script from a topic using SiliconFlow LLM."""
    words_per_second = 2.5
    target_words = int(req.duration_seconds * words_per_second)
    tone_desc = TONE_PROMPTS.get(req.tone, TONE_PROMPTS["professional"])

    prompt = (
        f"Write a natural spoken-word script about: \"{req.topic}\".\n"
        f"Requirements:\n"
        f"- Tone: {tone_desc}\n"
        f"- Length: approximately {target_words} words (for a {req.duration_seconds}-second video)\n"
        f"- Language: {req.language}\n"
        f"- NO stage directions, NO [brackets], NO speaker labels\n"
        f"- Write ONLY the words the avatar will say, nothing else\n"
        f"- Start speaking immediately, no intro like 'Sure, here is...'"
    )

    # Use Groq LLM for script generation
    providers: list[tuple[str, str, str]] = []
    if GROQ_KEY:
        providers.append((GROQ_BASE, GROQ_KEY, "llama-3.1-8b-instant"))

    last_err = ""
    for base_url, api_key, model in providers:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 600,
                        "temperature": 0.7,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                script = data["choices"][0]["message"]["content"].strip()
                word_count = len(script.split())
                return ScriptResponse(
                    script=script,
                    word_count=word_count,
                    estimated_duration=int(word_count / words_per_second),
                )
        except Exception as e:
            last_err = getattr(getattr(e, 'response', None), 'text', str(e))[:150]
            logger.warning(f"Script model={model} failed: {last_err}")
            await asyncio.sleep(0.5)

    # Final fallback: template-based script (no API needed)
    logger.warning("All LLM providers failed — using template fallback")
    script = _template_script(req.topic, req.tone, target_words)
    word_count = len(script.split())
    return ScriptResponse(script=script, word_count=word_count, estimated_duration=int(word_count / words_per_second))


# ── TTS Preview ───────────────────────────────────────────────────────────────

class TTSPreviewRequest(BaseModel):
    text: str = "Hello! This is a preview of how your avatar will sound."
    voice_id: str = "fishaudio/fish-speech-1.5:alex"

class TTSPreviewResponse(BaseModel):
    audio_url: str
    voice_id: str

_EDGE_VOICE_MAP = {
    "alex":  "en-US-GuyNeural",
    "sarah": "en-US-JennyNeural",
    "anna":  "en-GB-SoniaNeural",
    "bob":   "en-US-ChristopherNeural",
    "luna":  "en-US-AriaNeural",
}

async def _edge_tts_bytes(text: str, voice_id: str) -> bytes:
    """Generate MP3 bytes using Microsoft Edge TTS (no API key needed)."""
    import edge_tts, io
    short_id = voice_id.split(":")[-1] if ":" in voice_id else voice_id
    edge_voice = _EDGE_VOICE_MAP.get(short_id, "en-US-JennyNeural")
    communicate = edge_tts.Communicate(text, edge_voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return buf.read()

@router.post("/tts-preview", response_model=TTSPreviewResponse)
@limiter.limit("15/minute")
async def tts_preview(req: TTSPreviewRequest, request: Request):
    """Generate a short audio preview for a voice selection using edge-tts (free)."""
    preview_text = req.text[:200]  # cap at 200 chars for speed

    try:
        logger.info("Using edge-tts for TTS preview")
        audio_bytes = await _edge_tts_bytes(preview_text, req.voice_id)
        if audio_bytes:
            audio_url = _upload_r2(audio_bytes, ".mp3", "audio/mpeg")
            return TTSPreviewResponse(audio_url=audio_url, voice_id=req.voice_id)
    except ImportError:
        logger.error("edge-tts not installed")
        raise HTTPException(status_code=501, detail="TTS service unavailable. edge-tts not installed.")
    except Exception as e:
        logger.error(f"edge-tts failed: {e}")
        raise HTTPException(status_code=502, detail=f"TTS preview failed: {str(e)}")

    raise HTTPException(status_code=502, detail="TTS preview temporarily unavailable. Please try again.")


# ── Background Removal ────────────────────────────────────────────────────────

class RemoveBgResponse(BaseModel):
    url: str
    original_size: int
    processed_size: int

@router.post("/remove-bg", response_model=RemoveBgResponse)
@limiter.limit("10/minute")
async def remove_background(request: Request, file: UploadFile = File(...)):
    """
    Remove background from an uploaded portrait using rembg.
    Returns a presigned R2 URL to the transparent PNG.
    """
    try:
        from rembg import remove as rembg_remove
        from PIL import Image

        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Remove background
        img_input = Image.open(io.BytesIO(content)).convert("RGBA")
        img_output = rembg_remove(img_input)

        # Convert to PNG bytes
        buf = io.BytesIO()
        img_output.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        url = _upload_r2(png_bytes, ".png", "image/png")
        return RemoveBgResponse(
            url=url,
            original_size=len(content),
            processed_size=len(png_bytes),
        )

    except ImportError:
        raise HTTPException(status_code=501, detail="rembg not installed on this server")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Background removal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Brand Watermark ───────────────────────────────────────────────────────────

class WatermarkRequest(BaseModel):
    video_url: str
    company_name: str = ""
    logo_url: Optional[str] = None
    brand_color: str = "#8B5CF6"   # hex color for text
    position: str = "bottomright"  # bottomright | bottomleft | topright | topleft

class WatermarkResponse(BaseModel):
    video_url: str
    original_url: str

def _hex_to_ffmpeg_color(hex_color: str) -> str:
    """Convert #RRGGBB to ffmpeg color string."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return f"0x{h}FF"
    return "0x8B5CF6FF"

def _position_to_xy(position: str, logo_w: int = 80, margin: int = 16) -> tuple[str, str]:
    positions = {
        "bottomright": (f"W-{logo_w + margin}", f"H-{logo_w + margin}"),
        "bottomleft":  (str(margin), f"H-{logo_w + margin}"),
        "topright":    (f"W-{logo_w + margin}", str(margin)),
        "topleft":     (str(margin), str(margin)),
    }
    return positions.get(position, positions["bottomright"])

@router.post("/watermark", response_model=WatermarkResponse)
async def apply_watermark(req: WatermarkRequest):
    """Apply brand logo and/or company name watermark to a video."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_path = tmp / "input.mp4"
        output_path = tmp / "watermarked.mp4"

        # Download video
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(req.video_url)
            r.raise_for_status()
            video_path.write_bytes(r.content)

            # Download logo if provided
            logo_path: Optional[Path] = None
            if req.logo_url:
                try:
                    lr = await client.get(req.logo_url)
                    lr.raise_for_status()
                    ext = ".png" if "png" in lr.headers.get("content-type","") else ".jpg"
                    logo_path = tmp / f"logo{ext}"
                    logo_path.write_bytes(lr.content)
                except Exception as e:
                    logger.warning(f"Logo download failed, skipping: {e}")

        # Build FFmpeg filter
        ffmpeg_color = _hex_to_ffmpeg_color(req.brand_color)
        margin = 16
        x, y = _position_to_xy(req.position)

        if logo_path and req.company_name:
            filter_complex = (
                f"[1:v]scale=80:80[logo];"
                f"[0:v][logo]overlay={x}:{y}[tmp];"
                f"[tmp]drawtext=text='{req.company_name}':fontcolor={ffmpeg_color}:fontsize=18:"
                f"x={x}:y={y}-24:shadowcolor=black:shadowx=1:shadowy=1[out]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(logo_path),
                "-filter_complex", filter_complex,
                "-map", "[out]", "-map", "0:a?",
                "-c:v", "libx264", "-c:a", "copy", "-movflags", "+faststart",
                str(output_path)
            ]
        elif logo_path:
            filter_complex = f"[1:v]scale=80:80[logo];[0:v][logo]overlay={x}:{y}[out]"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path), "-i", str(logo_path),
                "-filter_complex", filter_complex,
                "-map", "[out]", "-map", "0:a?",
                "-c:v", "libx264", "-c:a", "copy", "-movflags", "+faststart",
                str(output_path)
            ]
        elif req.company_name:
            filter_complex = (
                f"drawtext=text='{req.company_name}':fontcolor={ffmpeg_color}:fontsize=18:"
                f"x={x}:y={y}:shadowcolor=black:shadowx=1:shadowy=1"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", filter_complex,
                "-c:v", "libx264", "-c:a", "copy", "-movflags", "+faststart",
                str(output_path)
            ]
        else:
            raise HTTPException(status_code=400, detail="Provide at least company_name or logo_url")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise HTTPException(status_code=500, detail="FFmpeg watermark failed")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="FFmpeg timed out")

        watermarked_bytes = output_path.read_bytes()
        new_url = _upload_r2(watermarked_bytes, ".mp4", "video/mp4")
        return WatermarkResponse(video_url=new_url, original_url=req.video_url)


# ── Video Concat ──────────────────────────────────────────────────────────────

class ConcatRequest(BaseModel):
    video_urls: list[str]  # ordered list of video URLs to join

class ConcatResponse(BaseModel):
    video_url: str

@router.post("/concat", response_model=ConcatResponse)
async def concat_videos(req: ConcatRequest):
    """Concatenate multiple videos into one using FFmpeg."""
    if len(req.video_urls) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 videos")
    if len(req.video_urls) > 6:
        raise HTTPException(status_code=400, detail="Max 6 videos")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_paths = []

        async with httpx.AsyncClient(timeout=60) as client:
            for i, url in enumerate(req.video_urls):
                r = await client.get(url)
                r.raise_for_status()
                p = tmp / f"clip_{i}.mp4"
                p.write_bytes(r.content)
                video_paths.append(p)

        # Write concat list file
        list_file = tmp / "list.txt"
        list_file.write_text("\n".join(f"file '{p}'" for p in video_paths))

        output_path = tmp / "concat.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                logger.error(f"FFmpeg concat error: {result.stderr}")
                raise HTTPException(status_code=500, detail="Concat failed")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Concat timed out")

        url = _upload_r2(output_path.read_bytes(), ".mp4", "video/mp4")
        return ConcatResponse(video_url=url)


# ── Auto-Captions ─────────────────────────────────────────────────────────────

class CaptionsRequest(BaseModel):
    video_url: str
    text: str
    language: str = "en"
    font_size: int = 22
    position: str = "bottom"   # bottom | top | center

class CaptionsResponse(BaseModel):
    video_url: str
    srt_url: str

def _text_to_srt(text: str, words_per_second: float = 2.3) -> str:
    """Generate a simple SRT file from text using word-count timing."""
    words = text.split()
    lines, chunk, t = [], [], 0.0
    for w in words:
        chunk.append(w)
        if len(chunk) >= 8 or w.endswith(('.', '!', '?', ',', ';')):
            lines.append((t, t + len(chunk) / words_per_second, ' '.join(chunk)))
            t += len(chunk) / words_per_second
            chunk = []
    if chunk:
        lines.append((t, t + len(chunk) / words_per_second, ' '.join(chunk)))

    def fmt(s: float) -> str:
        h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}".replace('.', ',')

    return "\n\n".join(f"{i+1}\n{fmt(s)} --> {fmt(e)}\n{t}" for i, (s, e, t) in enumerate(lines))

@router.post("/add-captions", response_model=CaptionsResponse)
@limiter.limit("10/minute")
async def add_captions(req: CaptionsRequest, request: Request):
    """Burn subtitles into a video using FFmpeg drawtext from text timing."""
    srt_content = _text_to_srt(req.text)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_path = tmp / "input.mp4"
        srt_path = tmp / "captions.srt"
        output_path = tmp / "captioned.mp4"

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(req.video_url)
            r.raise_for_status()
            video_path.write_bytes(r.content)

        srt_path.write_text(srt_content, encoding="utf-8")

        # Position mapping
        y_pos = {"bottom": "H-80", "top": "60", "center": "(H-th)/2"}.get(req.position, "H-80")

        # subtitles filter with style
        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
        vf = (
            f"subtitles='{srt_escaped}':force_style='"
            f"Fontsize={req.font_size},PrimaryColour=&HFFFFFF&,"
            f"OutlineColour=&H000000&,Outline=2,Shadow=1,"
            f"Alignment=2,MarginV=30'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264", "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"FFmpeg captions error: {result.stderr}")
                raise HTTPException(status_code=500, detail="Caption burn failed")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="FFmpeg timed out")

        video_url = _upload_r2(output_path.read_bytes(), ".mp4", "video/mp4")
        srt_url = _upload_r2(srt_content.encode(), ".srt", "text/plain")
        return CaptionsResponse(video_url=video_url, srt_url=srt_url)
