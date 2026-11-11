"""
Avatar Service - Uses SiliconFlow API for SDXL/Flux generation
"""

import os
from typing import Optional, Dict, Any
import logging
from api.services.siliconflow_client import SiliconFlowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AvatarService:
    """Avatar generation service using SiliconFlow API"""
    
    def __init__(self):
        """Initialize avatar service with SiliconFlow client"""
        # Load API key from environment
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            logger.error("SILICONFLOW_API_KEY not found in environment - Avatar service will not work")
            self.siliconflow_client = None
        else:
            try:
                self.siliconflow_client = SiliconFlowClient(api_key=api_key)
                logger.info("AvatarService initialized with SiliconFlow")
            except Exception as e:
                logger.error(f"Failed to initialize SiliconFlow client: {e}")
                self.siliconflow_client = None
    
    async def generate_avatar(
        self,
        prompt: str,
        style: str = "realistic",
        width: int = 1024,
        height: int = 1024,
        user_id: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate avatar using SiliconFlow API
        
        Args:
            prompt: Avatar description
            style: Style preset (realistic, cartoon, anime, etc.)
            width: Image width
            height: Image height
            user_id: User ID for personalization
            negative_prompt: What to avoid in the image
            num_inference_steps: Quality steps (20-50)
            guidance_scale: Prompt adherence (1-20)
            seed: Random seed for reproducibility
            
        Returns:
            Dict with avatar URL and metadata
        """
        # Check if client is initialized
        if not self.siliconflow_client:
            return {
                "success": False,
                "error": "SiliconFlow client not initialized - check API key",
                "prompt": prompt
            }
        
        try:
            # Enhance prompt with style
            style_prompts = {
                "realistic": "professional portrait, photorealistic, detailed skin texture, studio lighting, 8k quality",
                "cartoon": "cartoon style, animated character, vibrant colors, clean lines, digital art",
                "anime": "anime style, manga character, Japanese animation, detailed eyes, vibrant",
                "cyberpunk": "cyberpunk style, neon lighting, futuristic, sci-fi, high tech, detailed",
                "fantasy": "fantasy art style, magical, ethereal, detailed, artistic, dramatic lighting",
                "corporate": "professional headshot, corporate style, business attire, neutral background",
                "portrait": "elegant portrait, soft lighting, shallow depth of field, professional photography",
                "oil_painting": "oil painting style, classical art, renaissance, masterpiece, brushstrokes visible",
                "watercolor": "watercolor painting, soft colors, artistic, flowing, delicate",
                "3d_render": "3D render, octane render, detailed, professional lighting, studio quality"
            }
            
            enhanced_prompt = f"{prompt}, {style_prompts.get(style, style_prompts['realistic'])}"
            
            # Build negative prompt
            default_negative = "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text"
            final_negative = f"{default_negative}, {negative_prompt}" if negative_prompt else default_negative
            
            logger.info(f"Generating avatar: {enhanced_prompt[:60]}...")
            
            # Call SiliconFlow API
            result = await self.siliconflow_client.generate_image(
                prompt=enhanced_prompt,
                negative_prompt=final_negative,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            if not result.get("success"):
                logger.error(f"Avatar generation failed: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "prompt": prompt
                }
            
            logger.info(f"Avatar generated successfully: {result.get('image_url')}")
            
            # SiliconFlow returns a direct image URL
            image_url = result.get("image_url")
            image_path = None  # No local path needed for cloud storage
            
            return {
                "success": True,
                "avatar_url": image_url,
                "local_path": image_path,
                "prompt": prompt,
                "style": style,
                "width": width,
                "height": height,
                "user_id": user_id,
                "inference_time": result.get("inference_time")
            }
            
        except Exception as e:
            logger.error(f"Avatar service error: {e}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt
            }
    
    async def generate_avatar_variations(
        self,
        base_prompt: str,
        num_variations: int = 4,
        style: str = "realistic"
    ) -> Dict[str, Any]:
        """
        Generate multiple avatar variations
        
        Args:
            base_prompt: Base description
            num_variations: Number of variations (1-8)
            style: Style preset
            
        Returns:
            Dict with list of avatar URLs
        """
        variations = []
        
        for i in range(num_variations):
            try:
                # Add variation seed
                variation_prompt = f"{base_prompt}, variation {i+1}, different angle, unique pose"
                
                result = await self.generate_avatar(
                    prompt=variation_prompt,
                    style=style,
                    seed=i + 1000  # Different seed for each
                )
                
                if result.get("success"):
                    variations.append(result)
                    
            except Exception as e:
                logger.error(f"Variation {i} failed: {e}")
                continue
        
        return {
            "success": len(variations) > 0,
            "variations": variations,
            "count": len(variations)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check GPU service health"""
        return await self.gpu_client.health_check()


# Global instance
avatar_service = AvatarService()
