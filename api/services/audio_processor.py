"""
Audio Processing Service
Splits long text into chunks, processes in parallel, stitches audio.
"""

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles long-form TTS by splitting, parallel processing, and stitching."""

    def __init__(self):
        self.client = None  # SiliconFlowClient module was removed
        self.max_chunk_size = 500

    def split_text(self, text: str) -> list:
        """Split text by paragraph boundaries, keeping chunks under max_chunk_size."""
        paragraphs = text.split('\n')
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 1 <= self.max_chunk_size:
                current = (current + ' ' + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks

    async def process_long_text(
        self,
        text: str,
        voice: str = "fishaudio:fish-speech-v1.5",
        language: str = "en",
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """Split long text, process chunks in parallel, stitch audio."""
        chunks = self.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")

        if len(chunks) == 1:
            return await self.client.generate_tts(text=chunks[0], voice=voice, speed=speed)

        # Process all chunks in parallel
        tasks = [
            self.client.generate_tts(text=chunk, voice=voice, speed=speed)
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks)

        # Check for failures
        for i, result in enumerate(results):
            if not result.get("success"):
                return {"success": False, "error": f"Chunk {i} failed: {result.get('error')}"}

        # Stitch audio buffers together
        audio_buffers = [r["audio_content"] for r in results]
        combined = b''.join(audio_buffers)

        return {
            "success": True,
            "audio_content": combined,
            "format": "mp3",
            "provider": "siliconflow_fishspeech",
            "chunks_processed": len(chunks)
        }


# Lazy instantiation - only create when needed to avoid startup failures
audio_processor = None

def get_audio_processor():
    """Get audio processor instance (lazy initialization)."""
    global audio_processor
    if audio_processor is None:
        try:
            audio_processor = AudioProcessor()
        except Exception as e:
            logger.warning(f"Failed to initialize AudioProcessor: {e}")
            return None
    return audio_processor
