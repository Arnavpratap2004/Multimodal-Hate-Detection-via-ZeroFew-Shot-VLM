"""
OpenRouter-based Vision-Language Model implementation.
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional, Literal

import httpx
from PIL import Image

from ..config import settings
from .base import BaseVLM, VLMOutput
from .prompts import VLM_ANALYSIS_PROMPT, VLM_QUICK_ANALYSIS_PROMPT, VLM_CODE_MIXED_PROMPT


class OpenRouterVLM(BaseVLM):
    """
    Vision-Language Model implementation using OpenRouter API.
    
    Supports multiple VLM backends including GPT-4o, Gemini, and Claude.
    """
    
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt_mode: Literal["standard", "quick", "code_mixed"] = "standard"
    ):
        """
        Initialize the OpenRouter VLM client.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o"). Uses config default if None.
            api_key: OpenRouter API key. Uses config default if None.
            prompt_mode: Which prompt template to use for analysis.
        """
        self.model = model or settings.vlm_model
        # Try multiple sources for API key: parameter > env var > settings
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.prompt_mode = prompt_mode
        
        # Select prompt based on mode
        self._prompts = {
            "standard": VLM_ANALYSIS_PROMPT,
            "quick": VLM_QUICK_ANALYSIS_PROMPT,
            "code_mixed": VLM_CODE_MIXED_PROMPT
        }
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY in your .env file or pass api_key parameter."
            )
    
    def _get_prompt(self) -> str:
        """Get the appropriate prompt based on mode."""
        return self._prompts.get(self.prompt_mode, VLM_ANALYSIS_PROMPT)
    
    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Encode image to base64 for API transmission.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Tuple of (base64_encoded_data, media_type)
            
        Raises:
            FileNotFoundError: If image doesn't exist.
            ValueError: If image format is not supported or file is too large.
        """
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        
        file_size = path.stat().st_size
        if file_size > self.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image too large: {file_size / 1024 / 1024:.1f}MB. "
                f"Maximum size: {self.MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB"
            )
        
        # Determine media type
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_types.get(suffix, "image/jpeg")
        
        # Read and encode
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        return image_data, media_type
    
    def _parse_json_response(self, content: str) -> dict:
        """
        Parse JSON from VLM response, handling various formats.
        
        Args:
            content: Raw response content from VLM.
            
        Returns:
            Parsed JSON dictionary.
            
        Raises:
            ValueError: If JSON cannot be parsed.
        """
        # Try direct JSON parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*"visual_description"[^{}]*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    continue
        
        # Last resort: try to find any JSON-like structure
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        
        raise ValueError(f"Could not parse JSON from response: {content[:500]}...")
    
    async def analyze_image(self, image_path: str) -> VLMOutput:
        """
        Analyze a meme image using the OpenRouter VLM API.
        
        Args:
            image_path: Path to the meme image file.
            
        Returns:
            VLMOutput containing structured analysis results.
        """
        # Encode image
        image_data, media_type = self._encode_image(image_path)
        
        # Construct message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": self._get_prompt()
                    }
                ]
            }
        ]
        
        # Make API request
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/multimodal-hate-detection",
                    "X-Title": "Multimodal Hate Detection Research"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,  # Low temperature for consistent analysis
                    "max_tokens": 1500
                }
            )
            
            if response.status_code != 200:
                error_detail = response.text
                raise RuntimeError(
                    f"VLM API request failed with status {response.status_code}: {error_detail}"
                )
            
            result = response.json()
        
        # Extract content from response
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected API response format: {result}") from e
        
        # Parse JSON response
        parsed = self._parse_json_response(content)
        
        # Validate and create output
        return VLMOutput(
            visual_description=parsed.get("visual_description", ""),
            ocr_text=parsed.get("ocr_text", ""),
            implicit_meaning=parsed.get("implicit_meaning", ""),
            target_group=parsed.get("target_group"),
            hate_risk_level=parsed.get("hate_risk_level")
        )
    
    async def health_check(self) -> bool:
        """
        Check if the OpenRouter API is accessible.
        
        Returns:
            True if the API is healthy and configured correctly.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception:
            return False
    
    def set_prompt_mode(self, mode: Literal["standard", "quick", "code_mixed"]) -> None:
        """
        Change the prompt mode for subsequent analyses.
        
        Args:
            mode: The prompt mode to use.
        """
        if mode not in self._prompts:
            raise ValueError(f"Invalid prompt mode: {mode}")
        self.prompt_mode = mode
