"""
Image Generation Service for Aura-PAI Platform
===============================================

Provides image generation capabilities using various AI services.
"""

import asyncio
import aiohttp
import logging
import base64
import io
from typing import Dict, List, Any, Optional
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from core.config import settings

logger = logging.getLogger(__name__)


class GeneratedImage:
    """Represents a generated image."""
    
    def __init__(self, image_data: bytes, prompt: str, model: str = "unknown", metadata: Dict[str, Any] = None):
        self.image_data = image_data
        self.prompt = prompt
        self.model = model
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()
        self.size = len(image_data)
    
    def to_base64(self) -> str:
        """Convert image data to base64 string."""
        return base64.b64encode(self.image_data).decode('utf-8')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model": self.model,
            "timestamp": self.timestamp,
            "size_bytes": self.size,
            "base64_data": self.to_base64(),
            "metadata": self.metadata
        }

class ImageGenerationService:
    """Service for generating images using various AI providers."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_providers = []
        
    async def initialize(self):
        """Initialize the image generation service."""
        logger.info("Initializing Image Generation Service...")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)  # 2 minutes for image generation
        )
        
        # Check available providers
        await self._check_available_providers()
        
        logger.info(f"Image Generation Service initialized with providers: {self.available_providers}")
    
    async def cleanup(self):
        """Clean up the service."""
        if self.session:
            await self.session.close()
        logger.info("Image Generation Service cleaned up")
    
    async def _check_available_providers(self):
        """Check which image generation providers are available."""
        self.available_providers = []
        
        # Check if we have OpenAI API key for DALL-E
        if settings.OPENAI_API_KEY:
            self.available_providers.append("dalle")
            logger.info("DALL-E provider available")
        
        # Check if we have Hugging Face API key
        if hasattr(settings, 'HUGGINGFACE_API_KEY') and settings.HUGGINGFACE_API_KEY:
            self.available_providers.append("huggingface")
            logger.info("Hugging Face provider available")
        
        # Add free providers (no API key needed) - simplified and fast
        self.available_providers.append("pollinations")  # Free API - fast and reliable
        self.available_providers.append("placeholder")   # Always available fallback
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on image generation service."""
        return {
            "status": "healthy",
            "available_providers": self.available_providers,
            "capabilities": {
                "text_to_image": len(self.available_providers) > 0,
                "image_editing": "dalle" in self.available_providers,
                "style_transfer": False
            }
        }
    
    async def generate_image(
        self,
        prompt: str,
        provider: str = None,
        size: str = "1024x1024",
        style: str = None,
        num_images: int = 1
    ) -> List[GeneratedImage]:
        """Generate images based on text prompt."""
        
        if not self.available_providers:
            raise RuntimeError("No image generation providers available")
        
        # Use first available provider if none specified
        if not provider or provider not in self.available_providers:
            provider = self.available_providers[0]
        
        logger.info(f"Generating image with {provider}: '{prompt[:50]}...'")
        
        try:
            if provider == "dalle":
                return await self._generate_with_dalle(prompt, size, num_images)
            elif provider == "huggingface":
                return await self._generate_with_huggingface(prompt, size, num_images)
            elif provider == "pollinations":
                return await self._generate_with_pollinations(prompt, size, num_images)
            elif provider == "placeholder":
                return await self._generate_placeholder(prompt, size, num_images)
            else:
                # Default to pollinations if unknown provider
                return await self._generate_with_pollinations(prompt, size, num_images)
                
        except Exception as e:
            logger.error(f"Image generation failed with {provider}: {e}")
            # Fallback to placeholder
            if provider != "placeholder":
                logger.info("Falling back to placeholder generation")
                return await self._generate_placeholder(prompt, size, num_images)
            raise
    
    async def generate_image_tool(self, *args, **kwargs):
        """Wrapper for tool invocation."""
        return await self.generate_image(*args, **kwargs)

    async def interpret_image_tool(self, *args, **kwargs):
        """Placeholder for image interpretation tool."""
        logger.warning("interpret_image_tool is not fully implemented.")
        return {"analysis": "Image interpretation is not yet available."}

    async def _generate_with_dalle(self, prompt: str, size: str, num_images: int) -> List[GeneratedImage]:
        """Generate images using OpenAI DALL-E."""
        try:
            import openai
            
            # Initialize OpenAI client
            client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Generate image
            response = await client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=min(num_images, 1)  # DALL-E 3 only supports 1 image at a time
            )
            
            results = []
            for image_data in response.data:
                # Download the image
                async with self.session.get(image_data.url) as img_response:
                    if img_response.status == 200:
                        image_bytes = await img_response.read()
                        generated_image = GeneratedImage(
                            image_data=image_bytes,
                            prompt=prompt,
                            model="dall-e-3",
                            metadata={
                                "size": size,
                                "revised_prompt": getattr(image_data, 'revised_prompt', prompt)
                            }
                        )
                        results.append(generated_image)
            
            return results
            
        except Exception as e:
            logger.error(f"DALL-E generation failed: {e}")
            raise
    
    async def _generate_with_pollinations(self, prompt: str, size: str, num_images: int) -> List[GeneratedImage]:
        """Generate images using Pollinations.ai (free API) - simplified and fast."""
        logger.info(f"Attempting Pollinations generation for: {prompt}")
        try:
            import urllib.parse
            
            # Simple URL encoding
            encoded_prompt = urllib.parse.quote(prompt)
            
            # Use simple size - Pollinations handles it well
            api_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"
            
            # Give it a reasonable time to respond before falling back
            timeout = aiohttp.ClientTimeout(total=20)
            async with self.session.get(api_url, timeout=timeout) as response:
                if response.status == 200:
                    image_bytes = await response.read()
                    logger.info(f"Successfully generated image, size: {len(image_bytes)} bytes")
                    generated_image = GeneratedImage(
                        image_data=image_bytes,
                        prompt=prompt,
                        model="pollinations-ai",
                        metadata={
                            "size": "512x512",
                            "provider": "pollinations",
                            "api_url": api_url
                        }
                    )
                    return [generated_image]
                else:
                    logger.error(f"Pollinations API returned status {response.status}")
                    raise Exception(f"Pollinations API failed with status {response.status}")
                    
        except asyncio.TimeoutError:
            logger.error("Pollinations API timeout - falling back to placeholder")
            return await self._generate_placeholder(prompt, size, num_images)
        except Exception as e:
            logger.error(f"Pollinations generation failed: {e} - falling back to placeholder")
            return await self._generate_placeholder(prompt, size, num_images)
    
    async def _generate_with_huggingface(self, prompt: str, size: str, num_images: int) -> List[GeneratedImage]:
        """Generate images using Hugging Face Inference API."""
        try:
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Use Stable Diffusion model on Hugging Face
            api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
            
            headers = {
                "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            results = []
            for i in range(min(num_images, 2)):  # Limit for free tier
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "num_inference_steps": 20,
                        "guidance_scale": 7.5
                    }
                }
                
                async with self.session.post(api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        generated_image = GeneratedImage(
                            image_data=image_bytes,
                            prompt=prompt,
                            model="stable-diffusion-v1.5",
                            metadata={
                                "size": size,
                                "provider": "huggingface",
                                "model": "runwayml/stable-diffusion-v1-5"
                            }
                        )
                        results.append(generated_image)
                    else:
                        error_text = await response.text()
                        logger.warning(f"Hugging Face API returned status {response.status}: {error_text}")
            
            if not results:
                raise Exception("No images generated from Hugging Face API")
                
            return results
            
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            raise

    async def detect_image_generation_request(self, message: str) -> bool:
        """Detect if a message is requesting image generation."""
        message_lower = message.lower()
        
        # Extended list of image generation trigger phrases
        generation_phrases = [
            # Direct generation commands
            "generate an image",
            "create an image", 
            "make an image",
            "draw an image",
            "generate image",
            "create image",
            "make image", 
            "draw image",
            "generate pic",
            "create pic",
            "make pic",
            "draw pic",
            "generate picture",
            "create picture",
            "make picture",
            "draw picture",
            "generate photo",
            "create photo",
            "make photo",
            
            # Natural language patterns
            "image of",
            "picture of", 
            "pic of",
            "photo of",
            "illustration of",
            "artwork of",
            "painting of",
            "drawing of",
            "sketch of",
            
            # Show/display patterns
            "show me an image",
            "show me a picture",
            "show me a pic",
            "show me a photo",
            "display an image",
            "display a picture",
            
            # Casual expressions
            "can you draw",
            "can you create",
            "can you generate",
            "please draw",
            "please create", 
            "please generate",
            "i want an image",
            "i want a picture",
            "i want a pic",
            "i need an image",
            "i need a picture",
            "i need a pic"
        ]
        
        return any(phrase in message_lower for phrase in generation_phrases)
    
    async def _generate_placeholder(self, prompt: str, size: str, num_images: int) -> List[GeneratedImage]:
        """Generate placeholder images quickly for testing."""
        logger.info(f"Generating placeholder image for: {prompt}")
        try:
            # Parse size
            try:
                width, height = map(int, size.split('x'))
            except:
                width, height = 512, 512
            
            # Create a simple colored rectangle with text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create image
            img = Image.new('RGB', (width, height), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                # Try to use a default font
                font = ImageFont.load_default()
            except:
                font = None
            
            text_lines = [
                "Generated Image",
                f"Prompt: {prompt[:30]}...",
                f"Size: {width}x{height}",
                "Provider: Placeholder"
            ]
            
            y_offset = 50
            for line in text_lines:
                draw.text((10, y_offset), line, fill='black', font=font)
                y_offset += 30
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            image_bytes = img_buffer.getvalue()
            
            generated_image = GeneratedImage(
                image_data=image_bytes,
                prompt=prompt,
                model="placeholder",
                metadata={
                    "size": f"{width}x{height}",
                    "provider": "placeholder",
                    "type": "test_image"
                }
            )
            
            logger.info(f"Generated placeholder image: {len(image_bytes)} bytes")
            return [generated_image]
            
        except Exception as e:
            logger.error(f"Placeholder generation failed: {e}")
            # Return minimal image data
            minimal_image = GeneratedImage(
                image_data=b"fake_image_data",
                prompt=prompt,
                model="minimal-placeholder",
                metadata={"error": str(e)}
            )
            return [minimal_image]
