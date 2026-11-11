"""
CloneFlow AI — Modal Serverless GPU Worker  (ALL-IN-ONE)
==========================================================
Six independent GPU functions for AI video, audio, and dubbing.

┌─────────────────────────────────────────────────────────────────────────────┐
│  FUNCTION 1 — process_job  (Talking Avatar)                                 │
│    Fish Speech 1.5  GPU local  voice clone + natural TTS, no API key needed │
│    SadTalker        GPU local  lip-sync, natural head motion                 │
│    MuseTalk         GPU local  high-quality lip-sync (alternative)         │
│    LivePortrait     GPU local  expression-rich rendering                     │
│    GFPGAN           GPU local  face enhancement / super-resolution           │
│    edge-tts         CPU free   fallback TTS (always works)                  │
│  Container image: ~4 GB (packages)  +  Modal Volume: ~25 GB (weights)      │
├─────────────────────────────────────────────────────────────────────────────┤
│  FUNCTION 2 — generate_image  (Flux.1-schnell portrait generator)           │
│    Flux.1-schnell   GPU local  Black Forest Labs, Apache 2.0                │
│    ~24 GB on disk, ~12 GB VRAM with CPU offload — fits on T4 (16 GB)        │
│    Container image: ~4 GB (packages)  +  Modal Volume: ~24 GB (weights)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  FUNCTION 3 — generate_video  (CogVideoX-5B text-to-video)                  │
│    CogVideoX-5B     GPU local  Apache 2.0, ~18 GB, 720×480, up to 8s        │
│    Container image: ~4 GB (packages)  +  Modal Volume: ~18 GB (weights)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  FUNCTION 4 — generate_hunyuan_video  (HunyuanVideo/HappyHorse)             │
│    HunyuanVideo     GPU local  Tencent, ~20 GB, 720×1280, up to 5s          │
│    Container image: ~4 GB (packages)  +  Modal Volume: ~20 GB (weights)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  FUNCTION 5 — dubbing_pipeline  (Full dubbing with translation)             │
│    Whisper large-v3 GPU local  transcription (any language)                 │
│    NLLB-200         GPU local  translation (100 languages)                  │
│    Fish Speech 1.5  GPU local  TTS with voice cloning                       │
│    Wav2Lip          GPU local  lip-sync to original video                    │
│  Container image: ~4 GB (shared with avatar)  +  Modal Volume: ~25 GB      │
├─────────────────────────────────────────────────────────────────────────────┤
│  FUNCTION 6 — stitch_long_video  (Long videos via clip stitching)           │
│    Splits prompt into N segments, generates clips, concatenates            │
│    Enables 3-5 minute videos from single prompt                            │
│    Container image: ~4 GB (shared with video gen)                              │
└─────────────────────────────────────────────────────────────────────────────┘

SIZE OVERVIEW
  Avatar container              ~4 GB    (packages only)
  Image-gen container            ~4 GB    (packages only)
  Video-gen container            ~4 GB    (packages only)
  Avatar weights  (Modal Volume) ~25 GB   (downloaded once, persisted)
  Flux weights  (Modal Volume)  ~24 GB   (downloaded once, persisted)
  CogVideoX-5B weights (Vol)    ~18 GB   (downloaded once, persisted)
  HunyuanVideo weights (Vol)    ~20 GB   (downloaded once, persisted)
  Total volume storage           ~87 GB   × $0.10/GB/mo = ~$8.70/mo

COST  (Modal T4 @ $1.80/hr, A10G @ $3.60/hr — charged per second of actual use)
  Talking avatar     ~2.5 min  →  ~$0.075 / video
  Image generation   ~60 sec   →  ~$0.030 / image
  Video (CogVideoX)  ~60 sec   →  ~$0.060 / clip  (A10G)
  Video (Hunyuan)    ~60 sec   →  ~$0.060 / clip  (A10G)
  Dubbing            ~3 min    →  ~$0.090 / video (T4)
  Long video (stitch) ~5 min    →  ~$0.300 / video (A10G)
  $30 free credit    ≈  400 avatar videos  OR  mix of all functions

SECRETS — CREATE IN MODAL DASHBOARD  (modal.com → Secrets)
  cloneflow-r2:
    R2_ACCOUNT_ID           <your Cloudflare account ID>
    R2_ACCESS_KEY_ID        <your R2 access key>
    R2_SECRET_ACCESS_KEY    <your R2 secret key>
    R2_BUCKET_NAME          aiclone
  cloneflow-fly:
    FLY_API_URL         https://ai-clone.fly.dev
    WORKER_SECRET          <shared secret for worker callbacks>
  cloneflow-ai:             (optional — only needed for SiliconFlow TTS/image fallbacks)
    SILICONFLOW_API_KEY     <your SiliconFlow key — leave empty if not using>

DEPLOY STEPS
  pip install modal
  modal token new                   ← follow the link, paste token
  modal deploy modal_worker.py      ← builds images, prints 6 URLs
  → set on Fly.io:
      MODAL_WORKER_URL        = trigger_job URL
      MODAL_IMGGEN_URL        = trigger_image_gen URL
      MODAL_VIDGEN_URL        = trigger_video_gen URL
      MODAL_HUNYUAN_URL       = trigger_hunyuan_video URL
      MODAL_DUBBING_URL       = trigger_dubbing URL
      MODAL_STITCH_URL        = trigger_stitch_video URL
"""

import os
import io
import asyncio
import tempfile
import subprocess
import logging
from typing import Optional

import modal
from pydantic import BaseModel

try:
    import torch
    
    # 1. Safety monkeypatch for torch.xpu in case diffusers accesses torch.xpu.empty_cache
    if not hasattr(torch, "xpu"):
        class _XPUModule:
            @staticmethod
            def empty_cache():
                pass
        torch.xpu = _XPUModule()

    # 2. Patch PyTorch infer_schema to handle string type annotations safely for diffusers CogVideoX
    try:
        import torch._library.infer_schema
        _orig_infer_schema = torch._library.infer_schema.infer_schema
        def _safe_infer_schema(func, *args, **kwargs):
            if hasattr(func, "__annotations__"):
                new_ann = {}
                for k, v in func.__annotations__.items():
                    if isinstance(v, str):
                        try:
                            import typing
                            ns = {
                                "torch": torch,
                                "typing": typing,
                                "Optional": typing.Optional,
                                "List": typing.List,
                                "Tuple": typing.Tuple,
                                "Union": typing.Union,
                                "bool": bool,
                                "int": int,
                                "float": float,
                                "str": str,
                                "tuple": tuple,
                                "list": list,
                                "dict": dict,
                                "None": None,
                            }
                            new_ann[k] = eval(v, ns)
                        except Exception:
                            new_ann[k] = v
                    else:
                        new_ann[k] = v
                func.__annotations__ = new_ann
            return _orig_infer_schema(func, *args, **kwargs)
        torch._library.infer_schema.infer_schema = _safe_infer_schema
    except Exception:
        pass
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cloneflow")

# ── Paths inside containers ─────────────────────────────────────────────────────
AVATAR_MODELS_DIR = "/models/avatar"
SADTALKER_DIR    = "/models/avatar/sadtalker"
LIVEPORTRAIT_DIR = "/models/avatar/liveportrait"
FISH_SPEECH_DIR  = "/models/avatar/fish-speech"
GFPGAN_CKPT      = "/models/avatar/gfpgan/GFPGANv1.4.pth"
MUSETALK_DIR     = "/models/avatar/musetalk"
WAV2LIP_DIR      = "/models/avatar/wav2lip"
NLLB_DIR         = "/models/avatar/nllb"
FLUX_MODEL_DIR   = "/models/flux"
COGVIDEO_MODEL_DIR = "/models/cogvideox"
HUNYUAN_MODEL_DIR = "/models/hunyuan"

# ══════════════════════════════════════════════════════════════════════════════
#  MODAL APP + SECRETS + VOLUMES
# ══════════════════════════════════════════════════════════════════════════════

app = modal.App("cloneflow-worker")

_required_secrets = [
    modal.Secret.from_name("cloneflow-r2"),
    modal.Secret.from_name("cloneflow-fly"),
]
_hf_secret = modal.Secret.from_name("HF_TOKEN")
secrets = _required_secrets  # cloneflow-ai is optional and not required for core functions

# Persistent volumes for large model weights (downloaded once, reused forever)
flux_volume  = modal.Volume.from_name("cloneflow-flux",  create_if_missing=True)
video_volume = modal.Volume.from_name("cloneflow-video", create_if_missing=True)
hunyuan_volume = modal.Volume.from_name("cloneflow-hunyuan", create_if_missing=True)
avatar_volume = modal.Volume.from_name("cloneflow-avatar", create_if_missing=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONTAINER IMAGE 1 — Talking Avatar
#  SadTalker + LivePortrait + Fish Speech 1.5 + GFPGAN + MuseTalk + Wav2Lip + NLLB + Whisper
#  Size: ~4 GB (packages only). Weights stored in Modal Volume (~25 GB, downloaded once).
# ══════════════════════════════════════════════════════════════════════════════

_pytorch_base = (
    modal.Image.debian_slim(python_version="3.10")
    .env({
        "DIFFUSERS_NO_CUSTOM_OPS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    })
    .apt_install(
        "git", "ffmpeg", "wget", "curl", "unzip",
        "libsm6", "libxext6", "libglib2.0-0", "libgl1-mesa-glx",
        "libsndfile1", "libgomp1", "build-essential",
    )
    .pip_install(
        "numpy<2",
        "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
        "fastapi", "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

avatar_image = (
    _pytorch_base
    # ── All Python packages ────────────────────────────────────────────────────
    .pip_install(
        "fastapi", "uvicorn", "httpx", "boto3", "edge-tts", "huggingface_hub",
        "face_alignment==1.3.5", "imageio==2.19.3", "imageio-ffmpeg==0.4.7",
        "librosa", "pydub", "scipy", "kornia", "tqdm", "yacs", "pyyaml",
        "joblib", "scikit-image", "basicsr", "facexlib", "gfpgan",
        "av", "safetensors", "numpy<2", "onnxruntime-gpu",
        "opencv-python-headless", "matplotlib", "Pillow",
        "openai-whisper",  # For transcription in dubbing
    )
    # ── No repos cloned here - will download to volume at runtime ───────────────
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONTAINER IMAGE 2 — Image Generation (Flux.1-schnell)
#  Packages only (~4 GB). Weights stored in Modal Volume (~24 GB, downloaded once).
# ══════════════════════════════════════════════════════════════════════════════

image_gen_image = (
    _pytorch_base
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
        "diffusers==0.30.0", "transformers==4.44.2", "accelerate",
        "sentencepiece", "protobuf",
        "bitsandbytes",   # for 8-bit T5 to fit on T4 16 GB VRAM
        "fastapi", "uvicorn", "httpx", "boto3", "huggingface_hub",
        "Pillow", "numpy<2", "safetensors",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

video_gen_image = (
    _pytorch_base
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
        "diffusers==0.32.1", "transformers==4.46.3", "accelerate",
        "sentencepiece", "protobuf", "bitsandbytes",
        "fastapi", "uvicorn", "httpx", "boto3", "huggingface_hub",
        "imageio", "imageio-ffmpeg", "Pillow", "numpy<2", "safetensors",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _r2():
    import boto3
    from botocore.client import Config
    account = (
        os.environ.get("R2_ACCOUNT_ID")
        or os.environ.get("CLOUDFLARE_R2_ACCOUNT_ID")
        or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        or ""
    )
    access_key = (
        os.environ.get("R2_ACCESS_KEY_ID")
        or os.environ.get("R2_ACCESS_KEY")          # Modal secret name variant
        or os.environ.get("CLOUDFLARE_R2_ACCESS_KEY_ID")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or ""
    )
    secret_key = (
        os.environ.get("R2_SECRET_ACCESS_KEY")
        or os.environ.get("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY")
        or ""
    )

    if not (account and access_key and secret_key):
        keys = [k for k in os.environ if any(x in k for x in ["R2", "CLOUDFLARE", "AWS"])]
        log.warning(f"R2 credentials incomplete. Available storage env keys: {keys}")
        return None

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_r2(data: bytes, key: str, content_type: str) -> str:
    r2 = _r2()
    bucket = os.environ.get("R2_BUCKET_NAME") or os.environ.get("CLOUDFLARE_R2_BUCKET_NAME") or "cloneflow-videos"
    if r2:
        try:
            r2.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
            public_url = os.environ.get("R2_PUBLIC_URL", "")
            if public_url:
                return f"{public_url.rstrip('/')}/{key}"
            return r2.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=604800
            )
        except Exception as e:
            log.error(f"R2 upload exception: {e}")
    
    log.warning(f"R2 storage fallback used for key: {key}")
    return f"https://ai-clone.fly.dev/storage/fallback/{key}"


def fetch(url: str) -> bytes:
    import httpx
    r = httpx.get(url, follow_redirects=True, timeout=60,
                  headers={"User-Agent": "CloneFlow/1.0"})
    r.raise_for_status()
    return r.content


def notify(job_id: str, status: str,
           video_url: str | None = None, error: str | None = None,
           progress: int | None = None):
    import httpx
    base = os.environ.get("FLY_API_URL", "https://ai-clone.fly.dev")
    secret = os.environ.get("WORKER_SECRET", "")
    if not secret:
        log.warning("WORKER_SECRET env var not set; callback may be rejected by backend")
    payload: dict = {"status": status}
    if video_url:
        payload["video_url"] = video_url
    if error:
        payload["error"] = error
    if progress is not None:
        payload["progress"] = progress
    headers = {"x-worker-secret": secret}
    try:
        r = httpx.post(f"{base}/api/jobs/{job_id}/update",
                       json=payload, headers=headers, timeout=15)
        log.info(f"notify job {job_id[:8]} status={status} progress={progress} -> HTTP {r.status_code}")
    except Exception as e:
        log.warning(f"notify failed for job {job_id[:8]}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — TEXT TO SPEECH
#  Priority: Fish Speech 1.5 LOCAL (T4 GPU) → edge-tts fallback (free, CPU)
#
#  Fish Speech runs entirely on your Modal T4 — no external API, no API key.
#  Voice cloning: pass a voice_ref_url (any audio sample) and Fish Speech
#  copies the speaker's voice characteristics into the generated speech.
# ══════════════════════════════════════════════════════════════════════════════

_EDGE_VOICE_MAP = {
    "alex":  "en-US-GuyNeural",
    "sarah": "en-US-JennyNeural",
    "anna":  "en-GB-SoniaNeural",
    "bob":   "en-US-ChristopherNeural",
    "luna":  "en-US-AriaNeural",
}

async def _edge_tts_fallback(text: str, voice_id: str) -> bytes:
    """Free Microsoft edge-tts — always available, no API key."""
    import edge_tts
    short = voice_id.split(":")[-1] if ":" in voice_id else voice_id
    voice = _EDGE_VOICE_MAP.get(short, "en-US-JennyNeural")
    communicate = edge_tts.Communicate(text[:500], voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _fish_speech_local(text: str, out_wav: str,
                       voice_ref_path: str | None = None) -> bool:
    """
    Fish Speech 1.5 local inference (two-stage pipeline on GPU).
    Stage 1: Text → semantic tokens  (LLaMA 0.5B model)
    Stage 2: Tokens → waveform       (VQGAN decoder)
    Returns True on success.
    """
    ckpt = f"{FISH_SPEECH_DIR}/checkpoints/fish-speech-1.5"
    codes_path = out_wav + ".codes.npy"

    # Stage 1 — LLaMA text-to-semantic
    cmd1 = [
        "python3", f"{FISH_SPEECH_DIR}/tools/llama/generate.py",
        "--text", text[:500],
        "--checkpoint-path", ckpt,
        "--output-path", codes_path,
        "--num-samples", "1",
    ]
    if voice_ref_path and os.path.exists(voice_ref_path):
        cmd1 += ["--prompt-speech", voice_ref_path]
        log.info("Fish Speech: using voice cloning from reference audio")
    else:
        log.info("Fish Speech: using default speaker")

    r1 = subprocess.run(
        cmd1, cwd=FISH_SPEECH_DIR, capture_output=True, text=True,
        timeout=120, env={**os.environ, "PYTHONPATH": FISH_SPEECH_DIR},
    )
    if r1.returncode != 0:
        log.warning(f"Fish Speech stage 1 failed: {r1.stderr[-300:]}")
        return False

    # Stage 2 — VQGAN decode codes → waveform
    cmd2 = [
        "python3", f"{FISH_SPEECH_DIR}/tools/vqgan/inference.py",
        "-i", codes_path,
        "-o", out_wav,
        "--checkpoint-path", ckpt,
    ]
    r2 = subprocess.run(
        cmd2, cwd=FISH_SPEECH_DIR, capture_output=True, text=True,
        timeout=60, env={**os.environ, "PYTHONPATH": FISH_SPEECH_DIR},
    )
    if r2.returncode != 0:
        log.warning(f"Fish Speech stage 2 failed: {r2.stderr[-300:]}")
        return False

    log.info(f"Fish Speech: local inference OK → {out_wav}")
    return True


async def tts(text: str, voice_id: str,
              voice_ref_url: str | None = None,
              tmp_dir: str = "/tmp") -> bytes:
    """
    Generate speech audio.
    1. Download voice reference from R2/URL (if provided for cloning)
    2. Run Fish Speech 1.5 locally on GPU
    3. Fall back to edge-tts if Fish Speech fails (always works, free)
    """
    import httpx
    out_wav = os.path.join(tmp_dir, "tts_fish.wav")
    ref_path: str | None = None

    # Download voice reference for cloning
    if voice_ref_url:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(voice_ref_url, follow_redirects=True)
                if r.status_code == 200:
                    ref_path = os.path.join(tmp_dir, "voice_ref.wav")
                    with open(ref_path, "wb") as f:
                        f.write(r.content)
        except Exception as e:
            log.warning(f"Voice ref download failed: {e}")

    # Try Fish Speech local GPU
    if _fish_speech_local(text, out_wav, ref_path):
        with open(out_wav, "rb") as f:
            return f.read()

    # Fallback: edge-tts (always works, no GPU needed)
    log.info("TTS: falling back to edge-tts")
    return await _edge_tts_fallback(text, voice_id)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2A — SADTALKER LIP-SYNC (always runs)
# ══════════════════════════════════════════════════════════════════════════════

def sadtalker(image_path: str, audio_path: str, out_dir: str,
              enhance: bool = True, size: int = 256) -> str:
    """
    Run SadTalker inference. Natural head motion (no --still).
    enhance=True runs GFPGAN face super-resolution.
    """
    cmd = [
        "python3", f"{SADTALKER_DIR}/inference.py",
        "--driven_audio", audio_path,
        "--source_image",  image_path,
        "--result_dir",    out_dir,
        "--preprocess",    "full",
        "--size",          str(size),
    ]
    if enhance:
        cmd += ["--enhancer", "gfpgan",
                "--gfpgan_path", GFPGAN_CKPT]

    env = {**os.environ, "PYTHONPATH": SADTALKER_DIR}
    res = subprocess.run(cmd, cwd=SADTALKER_DIR, capture_output=True,
                         text=True, timeout=300, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"SadTalker failed:\n{res.stderr[-500:]}")

    mp4s = sorted(
        [f for f in os.listdir(out_dir) if f.endswith(".mp4")],
        key=lambda x: os.path.getmtime(os.path.join(out_dir, x)),
        reverse=True,
    )
    if not mp4s:
        raise RuntimeError("SadTalker produced no output file")
    return os.path.join(out_dir, mp4s[0])


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2B — LIVEPORTRAIT CASCADE (only when model="liveportrait")
#  Uses SadTalker output as driving motion → LivePortrait renders higher quality
# ══════════════════════════════════════════════════════════════════════════════

def liveportrait(src_image: str, driving_video: str, out_dir: str) -> str:
    """
    LivePortrait cascade:
      source  = original user portrait (full quality)
      driving = SadTalker output  (provides lip & head motion)
      result  = LivePortrait-rendered video (photorealistic expressions)
    Gracefully falls back to SadTalker output if LivePortrait fails.
    """
    out_path = os.path.join(out_dir, "liveportrait_out.mp4")
    cmd = [
        "python3", f"{LIVEPORTRAIT_DIR}/inference.py",
        "--source", src_image,
        "--driving", driving_video,
        "--output-dir", out_dir,
        "--flag_do_crop",
        "--flag_remap_input",
    ]
    env = {**os.environ, "PYTHONPATH": LIVEPORTRAIT_DIR}
    res = subprocess.run(cmd, cwd=LIVEPORTRAIT_DIR, capture_output=True,
                         text=True, timeout=300, env=env)
    if res.returncode != 0 or not os.path.exists(out_path):
        log.warning(f"LivePortrait failed — using SadTalker output. "
                    f"stderr: {res.stderr[-200:]}")
        return driving_video   # fall back to SadTalker result

    log.info("LivePortrait cascade complete")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2C — MUSETALK LIP-SYNC (alternative to SadTalker)
#  Higher quality lip-sync, runs on T4 GPU
# ══════════════════════════════════════════════════════════════════════════════

def musetalk(image_path: str, audio_path: str, out_dir: str) -> str:
    """
    Run MuseTalk inference for high-quality lip-sync.
    Alternative to SadTalker — user can choose via model parameter.
    """
    out_path = os.path.join(out_dir, "musetalk_out.mp4")
    cmd = [
        "python3", f"{MUSETALK_DIR}/inference.py",
        "--source_image", image_path,
        "--driving_audio", audio_path,
        "--output_dir", out_dir,
    ]
    env = {**os.environ, "PYTHONPATH": MUSETALK_DIR}
    res = subprocess.run(cmd, cwd=MUSETALK_DIR, capture_output=True,
                         text=True, timeout=300, env=env)
    if res.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"MuseTalk failed:\n{res.stderr[-500:]}")
    log.info("MuseTalk lip-sync complete")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Download Avatar Weights to Volume (first run only)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_avatar_weights():
    """Clone repos and download all avatar model weights to volume if not present."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    # Clone Fish Speech repo
    if not os.path.exists(FISH_SPEECH_DIR):
        log.info("Cloning Fish Speech repo...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/fishaudio/fish-speech.git", FISH_SPEECH_DIR
        ], check=True)
        # Use --use-pep517 to avoid dependency resolution issues
        subprocess.run([
            "pip", "install", "-e", FISH_SPEECH_DIR, "-q", "--use-pep517"
        ], check=True)

    # Fish Speech 1.5 weights
    fish_dir = f"{FISH_SPEECH_DIR}/checkpoints/fish-speech-1.5"
    if not os.path.exists(fish_dir):
        log.info("Downloading Fish Speech 1.5 weights...")
        os.makedirs(f"{FISH_SPEECH_DIR}/checkpoints", exist_ok=True)
        snapshot_download('fishaudio/fish-speech-1.5',
                         local_dir=fish_dir,
                         ignore_patterns=['*.md','*.gitattributes'])

    # Clone SadTalker repo
    if not os.path.exists(SADTALKER_DIR):
        log.info("Cloning SadTalker repo...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/OpenTalker/SadTalker.git", SADTALKER_DIR
        ], check=True)
        subprocess.run([
            "pip", "install", "-r", f"{SADTALKER_DIR}/requirements.txt", "-q"
        ], check=True)

    # SadTalker weights
    if not os.path.exists(f"{SADTALKER_DIR}/checkpoints/SadTalker_V0.0.2_256.safetensors"):
        log.info("Downloading SadTalker weights...")
        os.makedirs(f"{SADTALKER_DIR}/checkpoints", exist_ok=True)
        subprocess.run([
            "wget", "-q", "-O", f"{SADTALKER_DIR}/checkpoints/SadTalker_V0.0.2_256.safetensors",
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors"
        ], check=True)
        subprocess.run([
            "wget", "-q", "-O", f"{SADTALKER_DIR}/checkpoints/SadTalker_V0.0.2_512.safetensors",
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors"
        ], check=True)
        subprocess.run([
            "wget", "-q", "-O", f"{SADTALKER_DIR}/checkpoints/mapping_00109-model.pth.tar",
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar"
        ], check=True)
        subprocess.run([
            "wget", "-q", "-O", f"{SADTALKER_DIR}/checkpoints/mapping_00229-model.pth.tar",
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar"
        ], check=True)

    # GFPGAN weights
    if not os.path.exists(GFPGAN_CKPT):
        log.info("Downloading GFPGAN weights...")
        os.makedirs(os.path.dirname(GFPGAN_CKPT), exist_ok=True)
        subprocess.run([
            "wget", "-q", "-O", GFPGAN_CKPT,
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        ], check=True)

    # Clone LivePortrait repo
    if not os.path.exists(LIVEPORTRAIT_DIR):
        log.info("Cloning LivePortrait repo...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/KwaiVGI/LivePortrait.git", LIVEPORTRAIT_DIR
        ], check=True)

    # LivePortrait weights
    if not os.path.exists(f"{LIVEPORTRAIT_DIR}/pretrained_weights"):
        log.info("Downloading LivePortrait weights...")
        snapshot_download('KwaiVGI/LivePortrait',
                         local_dir=f"{LIVEPORTRAIT_DIR}/pretrained_weights",
                         ignore_patterns=['*.md','*.txt'])

    # Clone MuseTalk repo
    if not os.path.exists(MUSETALK_DIR):
        log.info("Cloning MuseTalk repo...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/TMElyralab/MuseTalk.git", MUSETALK_DIR
        ], check=True)

    # MuseTalk weights
    if not os.path.exists(f"{MUSETALK_DIR}/checkpoints"):
        log.info("Downloading MuseTalk weights...")
        snapshot_download('TMElyralab/MuseTalk',
                         local_dir=f"{MUSETALK_DIR}/checkpoints",
                         ignore_patterns=['*.md','*.txt'])

    # Clone Wav2Lip repo
    if not os.path.exists(WAV2LIP_DIR):
        log.info("Cloning Wav2Lip repo...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Rudrabha/Wav2Lip.git", WAV2LIP_DIR
        ], check=True)

    # Wav2Lip weights
    if not os.path.exists(f"{WAV2LIP_DIR}/checkpoints/wav2lip.pth"):
        log.info("Downloading Wav2Lip weights...")
        os.makedirs(f"{WAV2LIP_DIR}/checkpoints", exist_ok=True)
        subprocess.run([
            "curl", "-L", "-o", f"{WAV2LIP_DIR}/checkpoints/wav2lip.pth",
            "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth"
        ], check=True)

    # NLLB-200 weights
    if not os.path.exists(f"{NLLB_DIR}/pytorch_model.bin"):
        log.info("Downloading NLLB-200 weights...")
        os.makedirs(NLLB_DIR, exist_ok=True)
        snapshot_download('facebook/nllb-200-distilled-600M',
                         local_dir=NLLB_DIR,
                         ignore_patterns=['*.md','*.txt'])

    log.info("All avatar repos and weights ready")


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 1 — Talking Avatar  (Fish Speech + SadTalker/MuseTalk + LivePortrait)
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=avatar_image,
    gpu="l4",
    timeout=600,
    secrets=secrets,
    volumes={AVATAR_MODELS_DIR: avatar_volume},
    retries=1,
    scaledown_window=60,
)
def process_job(
    job_id: str,
    user_id: str,
    image_url: str,
    text: str,
    voice_id: str = "alex",
    voice_ref_url: Optional[str] = None,
    model: str = "sadtalker",
    enhance_face: bool = True,
):
    """
    Pipeline:
      1. Download portrait
      2. Fish Speech 1.5 LOCAL TTS (voice clone if ref provided) → edge-tts fallback
      3. Lip-sync: SadTalker or MuseTalk (user choice)
      4. LivePortrait cascade (if model=liveportrait)
      5. Upload to R2 → notify Fly.io
    """
    log.info(f"=== Job {job_id[:8]} | model={model} ===")
    notify(job_id, "processing")

    # Reload volume to load weights (download if first run)
    avatar_volume.reload()
    _ensure_avatar_weights()

    try:
        with tempfile.TemporaryDirectory() as tmp:

            # 1. Portrait
            img_path = os.path.join(tmp, "portrait.jpg")
            with open(img_path, "wb") as f:
                f.write(fetch(image_url))

            # 2. TTS — Fish Speech local GPU
            audio_bytes = asyncio.run(tts(text, voice_id, voice_ref_url, tmp))
            if not audio_bytes:
                raise RuntimeError("TTS produced no audio")
            audio_path = os.path.join(tmp, "speech.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # 3. Lip-sync: SadTalker or MuseTalk
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            if model == "musetalk":
                video_path = musetalk(img_path, audio_path, out_dir)
            else:
                video_path = sadtalker(img_path, audio_path, out_dir,
                                       enhance=enhance_face)

            # 4. LivePortrait cascade (only when explicitly requested)
            if model == "liveportrait":
                video_path = liveportrait(img_path, video_path, out_dir)
            elif model not in ("sadtalker", "musetalk", "liveportrait"):
                log.info(f"Model '{model}' not installed — using SadTalker output")

            # 5. Upload + notify
            with open(video_path, "rb") as f:
                video_data = f.read()
            video_url = upload_r2(video_data,
                                  f"videos/{user_id}/{job_id}.mp4",
                                  "video/mp4")
            notify(job_id, "completed", video_url=video_url)
            log.info(f"Job {job_id[:8]} DONE — {len(video_data)//1024} KB")
            return {"job_id": job_id, "video_url": video_url}

    except Exception as e:
        log.error(f"Job {job_id[:8]} FAILED: {e}")
        notify(job_id, "failed", error=str(e))
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 2 — Image Generation  (Flux.1-schnell, local T4 GPU)
#
#  Weights (~24 GB) stored in Modal Volume — downloaded ONCE on first call,
#  then persisted forever. Subsequent calls load from Volume in ~5 s.
#
#  Memory strategy: enable_model_cpu_offload() → peak VRAM ~12 GB (fits T4).
#  Generation time: ~45-90 s @ 512×512 on T4.
#  Free? YES — covered by $30 credit (~1000 images before credit runs out).
#  Ongoing: ~$0.03/image.
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image_gen_image,
    gpu="l4",
    timeout=300,
    secrets=secrets,
    volumes={FLUX_MODEL_DIR: flux_volume},
    scaledown_window=120,
)
def generate_image(
    prompt: str,
    job_id: str,
    user_id: str,
    width: int = 768,
    height: int = 768,
    num_steps: int = 4,      # Flux Schnell needs only 4 steps
    seed: int = 42,
    style: str = "realistic",
):
    """
    Generate a portrait/avatar image using Flux.1-schnell locally on T4.
    Downloads model to Modal Volume on first run (~24 GB, one-time).
    """
    import torch
    if not hasattr(torch, "xpu"):
        class _XPUModule:
            @staticmethod
            def empty_cache():
                pass
        torch.xpu = _XPUModule()
    from diffusers import FluxPipeline
    from huggingface_hub import snapshot_download
    import json

    # Download weights to Volume on first run only
    marker = os.path.join(FLUX_MODEL_DIR, ".ready")
    if not os.path.exists(marker):
        log.info("Flux: downloading weights to Modal Volume (one-time ~24 GB)...")
        snapshot_download(
            "black-forest-labs/FLUX.1-schnell",
            local_dir=FLUX_MODEL_DIR,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        with open(marker, "w") as f:
            f.write("ok")
        flux_volume.commit()
        log.info("Flux: weights saved to Volume")

    log.info("Flux: loading pipeline on L4...")
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")  # Fits fully in L4 24GB VRAM!
    
    log.info("Flux: compiling transformer for max speed (JIT)...")
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune")

    # Style enhancement
    style_addons = {
        "realistic":  "professional portrait, photorealistic, studio lighting, 8k",
        "anime":      "anime style, vibrant colors, detailed, Japanese animation",
        "cartoon":    "cartoon style, digital art, clean lines, vibrant",
        "corporate":  "professional headshot, business attire, neutral background",
        "fantasy":    "fantasy art, magical, ethereal, dramatic lighting",
    }
    full_prompt = f"{prompt}, {style_addons.get(style, style_addons['realistic'])}"

    log.info(f"Flux: generating {width}x{height} — '{full_prompt[:60]}...'")
    generator = torch.Generator().manual_seed(seed)
    result = pipe(
        prompt=full_prompt,
        num_inference_steps=num_steps,
        width=width,
        height=height,
        generator=generator,
    )
    image = result.images[0]

    # Save and upload to R2
    import io as _io
    buf = _io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    key = f"images/{user_id}/{job_id}.png"
    image_url = upload_r2(buf.read(), key, "image/png")

    notify(job_id, "completed", video_url=image_url)
    log.info(f"Flux: image uploaded → {key}")
    return {"job_id": job_id, "image_url": image_url}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 3 — Video Generation  (CogVideoX-5B, maximum quality, A10G GPU)
#
#  CogVideoX-5B by Zhipu AI — best quality open-source text-to-video model.
#  Apache 2.0 license. 720×480 resolution, 49 frames (~8 seconds @ 8fps).
#  VRAM: ~18 GB with model_cpu_offload → runs on A10G (24 GB VRAM).
#  Quality: cinematic, photorealistic motion, far beyond ZeroScope.
#  Weights: ~18 GB stored in Modal Volume (downloaded once).
#  Generation time: ~60 s on A10G.  Cost: ~$0.04/clip.
#  Free? YES — $30 credit covers ~750 clips.
#
#  Modal auto-scales: zero containers when idle, spins up new ones on demand.
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=video_gen_image,
    gpu="a10g",              # 24 GB VRAM — required for CogVideoX-5B
    timeout=180,
    secrets=secrets,
    volumes={COGVIDEO_MODEL_DIR: video_volume},
    scaledown_window=60,     # release GPU quickly to save quota
    max_containers=2,        # prevent GPU quota exhaustion (limit 10)
)
def generate_video(
    prompt: str,
    job_id: str,
    user_id: str,
    num_frames: int = 48,
    num_steps: int = 30,
    guidance_scale: float = 6.0,
    seed: int = 42,
):
    # CogVideoX VAE requires (num_frames - 1) % 4 == 0 due to 3D causal downsampling
    k = max(2, round((num_frames - 1) / 4))
    num_frames = min(49, 4 * k + 1)

    import torch
    if not hasattr(torch, "xpu"):
        class _XPUModule:
            @staticmethod
            def empty_cache():
                pass
        torch.xpu = _XPUModule()
    from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
    from huggingface_hub import snapshot_download
    import numpy as np
    import imageio

    notify(job_id, "processing", progress=15)

    # Download weights to Volume on first run only
    marker = os.path.join(COGVIDEO_MODEL_DIR, ".ready")
    if not os.path.exists(marker):
        log.info("CogVideoX: downloading weights (~18 GB, one-time)...")
        snapshot_download(
            "THUDM/CogVideoX-5b",
            local_dir=COGVIDEO_MODEL_DIR,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        with open(marker, "w") as f:
            f.write("ok")
        video_volume.commit()
        log.info("CogVideoX: weights saved to Volume")

    log.info(f"CogVideoX-5B: loading pipeline on A10G VRAM for job {job_id[:8]}...")
    notify(job_id, "processing", progress=30)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    pipe = CogVideoXPipeline.from_pretrained(
        COGVIDEO_MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    notify(job_id, "processing", progress=50)
    log.info(f"CogVideoX: generating {num_frames} frames — '{prompt[:80]}...'")

    # Clear VRAM cache completely before inference pass
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    frames = output.frames[0]  # list of PIL images (720×480)

    # Export frames → MP4
    notify(job_id, "processing", progress=85)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name

    with imageio.get_writer(out_path, fps=8, format="FFMPEG",
                            codec="libx264", quality=9) as writer:
        for frame in frames:
            writer.append_data(np.array(frame))

    with open(out_path, "rb") as f:
        video_data = f.read()
    os.unlink(out_path)

    key = f"generated-videos/{user_id}/{job_id}.mp4"
    video_url = upload_r2(video_data, key, "video/mp4")

    notify(job_id, "completed", video_url=video_url)
    log.info(f"CogVideoX: video uploaded → {key}  ({len(video_data)//1024} KB)")
    return {"job_id": job_id, "video_url": video_url}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 4 — HunyuanVideo (HappyHorse alternative, A10G GPU)
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=video_gen_image,
    gpu="a10g",
    timeout=180,
    secrets=secrets,
    volumes={HUNYUAN_MODEL_DIR: hunyuan_volume},
    scaledown_window=60,     # release GPU quickly to save quota
    max_containers=2,        # prevent GPU quota exhaustion
)
def generate_hunyuan(
    job_id: str,
    user_id: str,
    prompt: str,
    num_frames: int = 129,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
):
    # HunyuanVideo generates 129 frames by default, which is huge and causes OOM on 24GB VRAM.
    k = max(2, round((num_frames - 1) / 4))
    # Cap frames at 33 instead of 49 to strictly prevent A10G OOM during attention
    num_frames = min(33, 4 * k + 1)

    import torch
    if not hasattr(torch, "xpu"):
        class _XPUModule:
            @staticmethod
            def empty_cache():
                pass
        torch.xpu = _XPUModule()
    from diffusers import HunyuanVideoPipeline
    from huggingface_hub import snapshot_download
    import numpy as np
    import imageio

    # Proactive garbage collection to clear any leaked tensors from previous failed container invocations
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    notify(job_id, "processing", progress=15)

    # Download weights to Volume on first run only
    marker = os.path.join(HUNYUAN_MODEL_DIR, ".ready")
    model_index = os.path.join(HUNYUAN_MODEL_DIR, "model_index.json")
    if not os.path.exists(marker) or not os.path.exists(model_index):
        log.info("HunyuanVideo: downloading weights (~20 GB, one-time)...")
        snapshot_download(
            "hunyuanvideo-community/HunyuanVideo",
            local_dir=HUNYUAN_MODEL_DIR,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        with open(marker, "w") as f:
            f.write("ok")
        hunyuan_volume.commit()
        log.info("HunyuanVideo: weights saved to Volume")

    log.info(f"HunyuanVideo: loading pipeline on A10G VRAM for job {job_id[:8]}...")
    notify(job_id, "processing", progress=30)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from diffusers import BitsAndBytesConfig, HunyuanVideoTransformer3DModel
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load transformer with bitsandbytes 8-bit
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        HUNYUAN_MODEL_DIR,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    )
    
    # Load pipeline with the quantized transformer
    pipe = HunyuanVideoPipeline.from_pretrained(
        HUNYUAN_MODEL_DIR,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    import gc
    notify(job_id, "processing", progress=50)
    log.info(f"HunyuanVideo: encoding prompt and generating {num_frames} frames — '{prompt[:80]}...'")

    # Encode prompt manually on CPU to avoid loading huge LLaMA-3/T5 to GPU
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        device="cpu",
    )
    
    # Move prompt embeddings to GPU
    prompt_embeds = prompt_embeds.to("cuda")
    if pooled_prompt_embeds is not None:
        pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")
    if prompt_attention_mask is not None:
        prompt_attention_mask = prompt_attention_mask.to("cuda")

    # Delete text encoders and tokenizers to free CPU RAM
    if hasattr(pipe, "text_encoder"):
        del pipe.text_encoder
        pipe.text_encoder = None
    if hasattr(pipe, "text_encoder_2"):
        del pipe.text_encoder_2
        pipe.text_encoder_2 = None
    if hasattr(pipe, "tokenizer"):
        del pipe.tokenizer
        pipe.tokenizer = None
    if hasattr(pipe, "tokenizer_2"):
        del pipe.tokenizer_2
        pipe.tokenizer_2 = None
        
    # Now that the huge text encoders are removed from the pipeline,
    # we can safely move the rest of the pipeline to CUDA to properly 
    # configure the `_execution_device` without OOMing the GPU.
    # The transformer is already on CUDA (via BitsAndBytes), and VAE will follow.
    # NOTE: We do NOT call enable_model_cpu_offload() here because bitsandbytes
    # 8-bit models cannot be hot-swapped between CPU/GPU via the offload hooks.
    # Manual text-encoder deletion above is our memory strategy instead.
    pipe.to("cuda")
    
    gc.collect()
    torch.cuda.empty_cache()

    # Force PyTorch SDPA for memory-efficient attention (2026 best practice)
    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.transformer.enable_xformers_memory_efficient_attention()
            log.info("HunyuanVideo: xformers memory-efficient attention enabled")
        except Exception:
            log.info("HunyuanVideo: xformers not available, using PyTorch SDPA (default)")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    try:
        output = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_videos_per_prompt=1,
            num_inference_steps=num_steps,
            num_frames=num_frames,
            height=320,
            width=512,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    except torch.cuda.OutOfMemoryError:
        log.error("HunyuanVideo: CUDA OOM — freeing memory and notifying backend")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        notify(job_id, "failed", error="GPU out of memory. Please try a shorter prompt or fewer frames.")
        return {"job_id": job_id, "error": "OOM"}
    frames = output.frames[0]

    # Clear memory (8-bit models can't be moved with .to(), so we just delete and GC)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Export frames → MP4
    notify(job_id, "processing", progress=85)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name

    with imageio.get_writer(out_path, fps=24, format="FFMPEG",
                            codec="libx264", quality=9) as writer:
        for frame in frames:
            writer.append_data(np.array(frame))

    with open(out_path, "rb") as f:
        video_data = f.read()
    os.unlink(out_path)

    key = f"generated-videos/{user_id}/{job_id}.mp4"
    video_url = upload_r2(video_data, key, "video/mp4")

    notify(job_id, "completed", video_url=video_url)
    log.info(f"HunyuanVideo: video uploaded → {key}  ({len(video_data)//1024} KB)")
    return {"job_id": job_id, "video_url": video_url}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 5 — Dubbing Pipeline (Whisper → NLLB → Fish Speech → Wav2Lip)
#  Full dubbing with translation and lip-sync
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=avatar_image,
    gpu="l4",
    timeout=1200,
    secrets=secrets,
    volumes={AVATAR_MODELS_DIR: avatar_volume},
    scaledown_window=120,
)
def dubbing_pipeline(
    job_id: str,
    user_id: str,
    video_url: str,
    target_language: str = "en",
    voice_id: str = "alex",
):
    """
    Full dubbing pipeline:
      1. Download video
      2. Whisper transcription (any language)
      3. NLLB-200 translation to target language
      4. Fish Speech TTS with cloned voice
      5. Wav2Lip lip-sync to original video
      6. Upload dubbed video to R2
    """
    log.info(f"=== Dubbing job {job_id[:8]} | target={target_language} ===")
    notify(job_id, "processing")

    # Reload volume to load weights (download if first run)
    avatar_volume.reload()
    _ensure_avatar_weights()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Download original video
            video_path = os.path.join(tmp, "original.mp4")
            with open(video_path, "wb") as f:
                f.write(fetch(video_url))

            # 2. Whisper transcription
            import whisper
            model = whisper.load_model("large-v3")
            result = model.transcribe(video_path)
            transcript = result["text"]
            log.info(f"Whisper transcription: {transcript[:100]}...")

            # 3. NLLB-200 translation
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tokenizer = AutoTokenizer.from_pretrained(f"{NLLB_DIR}")
            model = AutoModelForSeq2SeqLM.from_pretrained(f"{NLLB_DIR}")
            inputs = tokenizer(transcript, return_tensors="pt")
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_language])
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            log.info(f"Translated text: {translated_text[:100]}...")

            # 4. Fish Speech TTS
            audio_bytes = asyncio.run(tts(translated_text, voice_id, None, tmp))
            audio_path = os.path.join(tmp, "dubbed_audio.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # 5. Wav2Lip lip-sync
            dubbed_path = os.path.join(tmp, "dubbed.mp4")
            cmd = [
                "python3", f"{WAV2LIP_DIR}/inference.py",
                "--face", video_path,
                "--audio", audio_path,
                "--outfile", dubbed_path,
            ]
            env = {**os.environ, "PYTHONPATH": WAV2LIP_DIR}
            res = subprocess.run(cmd, cwd=WAV2LIP_DIR, capture_output=True,
                             text=True, timeout=600, env=env)
            if res.returncode != 0:
                raise RuntimeError(f"Wav2Lip failed:\n{res.stderr[-500:]}")

            # 6. Upload dubbed video
            with open(dubbed_path, "rb") as f:
                dubbed_data = f.read()
            key = f"dubbed-videos/{user_id}/{job_id}.mp4"
            dubbed_url = upload_r2(dubbed_data, key, "video/mp4")

            notify(job_id, "completed", video_url=dubbed_url)
            log.info(f"Dubbing complete → {key}  ({len(dubbed_data)//1024} KB)")
            return {"job_id": job_id, "video_url": dubbed_url}

    except Exception as e:
        log.error(f"Dubbing job {job_id[:8]} FAILED: {e}")
        notify(job_id, "failed", error=str(e))
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCTION 6 — Clip Stitching for Long Videos
#  Splits long prompt into multiple 10s clips, generates in parallel, concatenates
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=video_gen_image,
    gpu="a10g",
    timeout=1800,
    secrets=secrets,
    volumes={COGVIDEO_MODEL_DIR: video_volume},
    scaledown_window=120,
)
def stitch_long_video(
    prompt: str,
    job_id: str,
    user_id: str,
    target_duration: int = 60,  # target duration in seconds
    model: str = "cogvideox",  # cogvideox or hunyuan
    scene_prompts: Optional[list[str]] = None,
):
    """
    Generate a long video by stitching multiple 10s clips.
    1. Use Scene Director Agent `scene_prompts` when provided, else split prompt into segments
    2. Generate N clips sequentially on A10G
    3. Concatenate with ffmpeg
    4. Upload to R2
    """
    import torch
    import numpy as np
    import imageio
    from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
    from huggingface_hub import snapshot_download

    cleaned_scenes = [s.strip() for s in (scene_prompts or []) if isinstance(s, str) and s.strip()]
    if cleaned_scenes:
        segments = cleaned_scenes[:8]
    else:
        segments = []
        words = prompt.split()
        chunk_size = max(10, len(words) // max(1, target_duration // 10))
        for i in range(0, len(words), chunk_size):
            segments.append(" ".join(words[i:i+chunk_size]))

    log.info(f"Stitching {len(segments)} clips for {target_duration}s video")

    # Download weights once
    marker = os.path.join(COGVIDEO_MODEL_DIR, ".ready")
    if not os.path.exists(marker):
        log.info("CogVideoX: downloading weights (~18 GB, one-time)...")
        snapshot_download(
            "THUDM/CogVideoX-5b",
            local_dir=COGVIDEO_MODEL_DIR,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        with open(marker, "w") as f:
            f.write("ok")
        video_volume.commit()

    # Load pipeline once
    log.info("CogVideoX-5B: loading pipeline on A10G...")
    pipe = CogVideoXPipeline.from_pretrained(
        COGVIDEO_MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    # Generate clips in parallel
    clip_paths = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, seg_prompt in enumerate(segments):
            log.info(f"Generating clip {i+1}/{len(segments)}: '{seg_prompt[:50]}...'")
            generator = torch.Generator().manual_seed(42 + i)
            output = pipe(
                prompt=seg_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=30,  # faster for stitching
                num_frames=49,
                guidance_scale=6.0,
                generator=generator,
            )
            frames = output.frames[0]

            clip_path = os.path.join(tmp, f"clip_{i}.mp4")
            with imageio.get_writer(clip_path, fps=8, format="FFMPEG",
                                    codec="libx264", quality=9) as writer:
                for frame in frames:
                    writer.append_data(np.array(frame))
            clip_paths.append(clip_path)

        # Concatenate clips with ffmpeg
        concat_list = os.path.join(tmp, "concat.txt")
        with open(concat_list, "w") as f:
            for path in clip_paths:
                f.write(f"file '{path}'\n")

        final_path = os.path.join(tmp, "stitched.mp4")
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy", final_path,
            "-y"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {res.stderr[-500:]}")

        # Upload final video
        with open(final_path, "rb") as f:
            video_data = f.read()
        key = f"stitched-videos/{user_id}/{job_id}.mp4"
        video_url = upload_r2(video_data, key, "video/mp4")

        notify(job_id, "completed", video_url=video_url)
        log.info(f"Stitched video uploaded → {key}  ({len(video_data)//1024} KB)")
        return {"job_id": job_id, "video_url": video_url}


# ══════════════════════════════════════════════════════════════════════════════
#  TEST FUNCTION — Trigger weight download on first deployment
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=avatar_image,
    gpu="l4",
    timeout=1800,
    secrets=secrets,
    volumes={AVATAR_MODELS_DIR: avatar_volume},
)
def download_avatar_weights():
    """Download all avatar weights to volume (run once after deployment)."""
    log.info("Starting avatar weights download...")
    avatar_volume.reload()
    _ensure_avatar_weights()
    log.info("Avatar weights download complete!")
    return {"status": "success", "message": "All weights downloaded to volume"}


@app.function(
    image=image_gen_image,
    gpu="l4",
    timeout=1800,
    secrets=secrets + [_hf_secret],
    volumes={FLUX_MODEL_DIR: flux_volume},
)
def download_flux_weights():
    """Download Flux.1-schnell weights to volume (run once after deployment)."""
    import os
    from huggingface_hub import snapshot_download

    log.info("Starting Flux weights download...")
    flux_volume.reload()

    marker = os.path.join(FLUX_MODEL_DIR, ".ready")
    if not os.path.exists(marker):
        log.info("Flux: downloading weights to Modal Volume (one-time ~24 GB)...")
        snapshot_download(
            "black-forest-labs/FLUX.1-schnell",
            local_dir=FLUX_MODEL_DIR,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        with open(marker, "w") as f:
            f.write("ok")
        flux_volume.commit()
        log.info("Flux: weights saved to Volume")
    else:
        log.info("Flux weights already downloaded")

    return {"status": "success", "message": "Flux weights downloaded to volume"}


@app.function(
    image=video_gen_image,
    gpu="a10g",
    timeout=1800,
    secrets=secrets,
    volumes={COGVIDEO_MODEL_DIR: video_volume},
)
def download_cogvideox_weights():
    """Download CogVideoX-5B weights to volume (run once after deployment)."""
    import os
    from huggingface_hub import snapshot_download

    log.info("Starting CogVideoX weights download...")
    video_volume.reload()

    marker = os.path.join(COGVIDEO_MODEL_DIR, ".ready")
    if not os.path.exists(marker):
        log.info("CogVideoX: downloading weights to Modal Volume (one-time ~18 GB)...")
        snapshot_download(
            "THUDM/CogVideoX-5b",
            local_dir=COGVIDEO_MODEL_DIR,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        with open(marker, "w") as f:
            f.write("ok")
        video_volume.commit()
        log.info("CogVideoX: weights saved to Volume")
    else:
        log.info("CogVideoX weights already downloaded")

    return {"status": "success", "message": "CogVideoX weights downloaded to volume"}


@app.function(
    image=video_gen_image,
    gpu="a10g",
    timeout=1800,
    secrets=secrets,
    volumes={HUNYUAN_MODEL_DIR: hunyuan_volume},
)
def download_hunyuan_weights():
    """Download HunyuanVideo weights to volume (run once after deployment)."""
    import os
    from huggingface_hub import snapshot_download

    log.info("Starting HunyuanVideo weights download...")
    hunyuan_volume.reload()

    marker = os.path.join(HUNYUAN_MODEL_DIR, ".ready")
    model_index = os.path.join(HUNYUAN_MODEL_DIR, "model_index.json")
    if not os.path.exists(marker) or not os.path.exists(model_index):
        log.info("HunyuanVideo: downloading weights to Modal Volume (one-time ~20 GB)...")
        snapshot_download(
            "hunyuanvideo-community/HunyuanVideo",
            local_dir=HUNYUAN_MODEL_DIR,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        with open(marker, "w") as f:
            f.write("ok")
        hunyuan_volume.commit()
        log.info("HunyuanVideo: weights saved to Volume")
    else:
        log.info("HunyuanVideo weights already downloaded")

    return {"status": "success", "message": "HunyuanVideo weights downloaded to volume"}


# ══════════════════════════════════════════════════════════════════════════════
#  WEB ENDPOINTS — Fly.io calls these HTTP endpoints to trigger each function
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI

web_app = FastAPI()

class JobRequest(BaseModel):
    job_id: str
    user_id: str
    image_url: str
    text: str
    voice_id: str = "alex"
    voice_ref_url: Optional[str] = None
    model: str = "sadtalker"
    enhance_face: bool = True


class ImageGenRequest(BaseModel):
    job_id: str
    user_id: str
    prompt: str
    style: str = "realistic"
    width: int = 768
    height: int = 768
    seed: int = 42


class VideoGenRequest(BaseModel):
    job_id: str
    user_id: str
    prompt: str
    num_frames: int = 49
    num_steps: int = 50
    guidance_scale: float = 6.0
    seed: int = 42


class HunyuanVideoRequest(BaseModel):
    job_id: str
    user_id: str
    prompt: str
    num_frames: int = 129
    num_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42


class DubbingRequest(BaseModel):
    job_id: str
    user_id: str
    video_url: str
    target_language: str = "en"
    voice_id: str = "alex"
    voice_ref_url: Optional[str] = None


class StitchVideoRequest(BaseModel):
    job_id: str
    user_id: str
    prompt: str
    target_duration: int = 60
    model: str = "cogvideox"
    scene_prompts: Optional[list[str]] = None


@web_app.post("/trigger_job")
def trigger_job(body: JobRequest):
    """Fly.io calls this when user clicks Generate (talking avatar)."""
    process_job.spawn(**body.model_dump())
    return {"status": "accepted", "job_id": body.job_id}


@web_app.post("/trigger_image_gen")
def trigger_image_gen(body: ImageGenRequest):
    """Fly.io calls this when user clicks Generate AI Avatar (Flux Schnell)."""
    generate_image.spawn(**body.model_dump())
    return {"status": "accepted", "job_id": body.job_id}


@web_app.post("/trigger_video_gen")
def trigger_video_gen(body: VideoGenRequest):
    """Fly.io calls this when user requests a text-to-video clip (CogVideoX)."""
    generate_video.spawn(**body.model_dump())
    return {"status": "accepted", "job_id": body.job_id}


@web_app.post("/trigger_hunyuan_video")
def trigger_hunyuan_video(body: HunyuanVideoRequest):
    """Fly.io calls this when user requests HunyuanVideo (HappyHorse alternative)."""
    generate_hunyuan.spawn(**body.model_dump())
    return {"status": "accepted", "job_id": body.job_id}


@web_app.post("/trigger_dubbing")
def trigger_dubbing(body: DubbingRequest):
    """Fly.io calls this when user requests full dubbing with translation and lip-sync."""
    dubbing_pipeline.spawn(**body.model_dump())
    return {"status": "accepted", "job_id": body.job_id}


@web_app.post("/trigger_stitch_video")
def trigger_stitch_video(body: StitchVideoRequest):
    """Fly.io calls this when user requests a long video via clip stitching."""
    stitch_long_video.spawn(**body.model_dump())
    return {"status": "accepted", "job_id": body.job_id}


@web_app.post("/trigger_download")
def trigger_download():
    """Trigger background download of all model weights to Modal Volumes."""
    download_avatar_weights.spawn()
    download_flux_weights.spawn()
    download_cogvideox_weights.spawn()
    download_hunyuan_weights.spawn()
    return {
        "status": "accepted",
        "message": "All model weight downloads (Avatar, Flux, CogVideoX, Hunyuan) triggered in parallel on Modal Volumes"
    }


@web_app.get("/health")
def health():
    return {
        "status": "ok",
        "worker": "cloneflow-modal",
        "functions": {
            "process_job":        "SadTalker/MuseTalk+LivePortrait+FishSpeech — T4",
            "generate_image":     "Flux.1-schnell — T4",
            "generate_video":     "CogVideoX-5B — A10G (720×480, 8s)",
            "generate_hunyuan":   "HunyuanVideo (HappyHorse) — A10G (720×1280, 5s)",
            "dubbing_pipeline":  "Whisper+NLLB+FishSpeech+Wav2Lip — T4",
            "stitch_long_video":  "Clip stitching for long videos (up to 3-5 min) — A10G",
        },
    }


web_image = modal.Image.debian_slim().pip_install("fastapi", "pydantic", "httpx")

@app.function(image=web_image)
@modal.asgi_app()
def web():
    """Serve the FastAPI web app for HTTP endpoints."""
    return web_app
