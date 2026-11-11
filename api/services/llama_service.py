"""
Llama Service - Connects to RunPod GPU for text generation
"""

import os
from typing import Optional, Dict, Any
import logging
from api.services.gpu_client import RunPodGPUClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaService:
    """AI script generation service using Llama 3 on RunPod GPU"""
    
    def __init__(self):
        """Initialize Llama service with GPU client"""
        gpu_url = os.getenv("RUNPOD_GPU_URL", "http://localhost:8000")
        self.gpu_client = RunPodGPUClient(base_url=gpu_url)
        logger.info(f"LlamaService initialized with GPU at: {gpu_url}")
    
    async def generate_script(
        self,
        topic: str,
        tone: str = "professional",
        length: str = "medium",
        target_audience: Optional[str] = None,
        key_points: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate video script using Llama 3
        
        Args:
            topic: Video topic
            tone: Tone of voice (professional, casual, enthusiastic, etc.)
            length: Script length (short, medium, long)
            target_audience: Who the video is for
            key_points: Key points to cover
            
        Returns:
            Dict with generated script
        """
        try:
            # Build system prompt
            system_prompt = f"""You are an expert video script writer. Create engaging, well-structured video scripts.
Tone: {tone}
Target Audience: {target_audience or "general audience"}

Structure:
1. Hook (attention-grabbing opening)
2. Introduction (what the video is about)
3. Main Content (key points)
4. Conclusion (summary)
5. Call to Action (what viewers should do next)

Write in a conversational, natural style suitable for video narration."""

            # Build user prompt
            length_words = {
                "short": "200-300 words, 1-2 minutes",
                "medium": "400-600 words, 3-5 minutes",
                "long": "800-1000 words, 7-10 minutes"
            }
            
            user_prompt = f"""Write a {length} video script about: {topic}

Target Length: {length_words.get(length, "400-600 words")}

"""
            
            if key_points:
                user_prompt += "Key Points to Cover:\n"
                for point in key_points:
                    user_prompt += f"- {point}\n"
                user_prompt += "\n"
            
            user_prompt += "Script:"
            
            logger.info(f"Generating script for topic: {topic[:50]}...")
            
            # Call GPU service
            result = await self.gpu_client.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            if not result.get("success"):
                logger.error(f"Script generation failed: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "topic": topic
                }
            
            script_text = result.get("text", "")
            word_count = len(script_text.split())
            
            logger.info(f"Script generated: {word_count} words")
            
            return {
                "success": True,
                "script": script_text,
                "topic": topic,
                "tone": tone,
                "length": length,
                "word_count": word_count,
                "estimated_duration": self._estimate_duration(word_count),
                "tokens_generated": result.get("tokens_generated"),
                "inference_time": result.get("inference_time")
            }
            
        except Exception as e:
            logger.error(f"Script generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }
    
    async def improve_script(
        self,
        script: str,
        feedback: str,
        focus_areas: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Improve existing script based on feedback
        
        Args:
            script: Original script
            feedback: What to improve
            focus_areas: Specific areas to focus on
            
        Returns:
            Dict with improved script
        """
        try:
            system_prompt = "You are an expert script editor. Improve the provided script based on feedback while maintaining the original message and structure."
            
            focus_text = ""
            if focus_areas:
                focus_text = f"\nFocus Areas: {', '.join(focus_areas)}\n"
            
            prompt = f"""Improve this video script based on the feedback:

FEEDBACK:
{feedback}{focus_text}

ORIGINAL SCRIPT:
{script}

IMPROVED SCRIPT:
"""
            
            result = await self.gpu_client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1200,
                temperature=0.6
            )
            
            if not result.get("success"):
                return result
            
            improved_script = result.get("text", "")
            
            return {
                "success": True,
                "original_script": script,
                "improved_script": improved_script,
                "feedback": feedback,
                "improvements_made": focus_areas or ["general improvement"]
            }
            
        except Exception as e:
            logger.error(f"Script improvement error: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_social_content(
        self,
        script: str,
        platforms: list = None
    ) -> Dict[str, Any]:
        """
        Generate social media posts from video script
        
        Args:
            script: Video script
            platforms: Social platforms (twitter, linkedin, instagram, etc.)
            
        Returns:
            Dict with social media posts
        """
        try:
            platforms = platforms or ["twitter", "linkedin"]
            
            system_prompt = "You are a social media expert. Create engaging posts from video scripts."
            
            prompt = f"""Create social media posts from this video script:

SCRIPT:
{script[:1000]}...

PLATFORMS: {', '.join(platforms)}

Generate separate posts for each platform with appropriate hashtags and formatting.

SOCIAL MEDIA POSTS:"""
            
            result = await self.gpu_client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=800,
                temperature=0.8
            )
            
            if not result.get("success"):
                return result
            
            return {
                "success": True,
                "social_posts": result.get("text"),
                "platforms": platforms
            }
            
        except Exception as e:
            logger.error(f"Social content error: {e}")
            return {"success": False, "error": str(e)}
    
    def _estimate_duration(self, word_count: int) -> str:
        """Estimate video duration from word count"""
        # Average speaking rate: 130-150 words per minute
        minutes = word_count / 140
        
        if minutes < 1:
            return f"{int(minutes * 60)} seconds"
        elif minutes < 60:
            return f"{int(minutes)} minutes"
        else:
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return f"{hours} hour{'s' if hours > 1 else ''} {mins} min"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check GPU service health"""
        return await self.gpu_client.health_check()


# Global instance
llama_service = LlamaService()
