"""
Ollama-based local Vision-Language Model implementation.
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional, Literal

import httpx

from .base import BaseVLM, VLMOutput
from .prompts import VLM_ANALYSIS_PROMPT, VLM_QUICK_ANALYSIS_PROMPT, VLM_CODE_MIXED_PROMPT


class OllamaVLM(BaseVLM):
    """
    Vision-Language Model implementation using local Ollama server.
    
    Uses LLaVA model for image analysis.
    """
    
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB for local
    
    def __init__(
        self,
        model: str = "llava:7b",
        base_url: str = "http://localhost:11434",
        prompt_mode: Literal["standard", "quick", "code_mixed"] = "standard"
    ):
        """
        Initialize the Ollama VLM client.
        
        Args:
            model: Ollama model name (default: llava:7b)
            base_url: Ollama API base URL
            prompt_mode: Which prompt template to use
        """
        self.model = model
        self.base_url = base_url
        self.prompt_mode = prompt_mode
        
        self._prompts = {
            "standard": VLM_ANALYSIS_PROMPT,
            "quick": VLM_QUICK_ANALYSIS_PROMPT,
            "code_mixed": VLM_CODE_MIXED_PROMPT
        }
        
        # Simplified prompt for LLaVA (local models)
        self._llava_prompt = """Analyze this meme image:

1. Describe what you see in the image (people, objects, expressions).
2. What text appears in the image? Copy it exactly.
3. What is the message or meaning of this meme?
4. Could this content be considered offensive, hateful, or inappropriate? Explain why or why not.
5. Who might be targeted or affected by this content?

Be thorough but concise."""
    
    def _get_prompt(self) -> str:
        """Get simplified prompt for LLaVA."""
        # Use simpler prompt for better LLaVA compatibility
        return self._llava_prompt
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}")
        
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from VLM response with robust fallback."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try markdown code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*"visual_description"[^{}]*\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    continue
        
        # Try to find any JSON structure
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        
        # FALLBACK: Extract from free-form text
        return self._extract_from_text(content)
    
    def _extract_from_text(self, content: str) -> dict:
        """Extract structured data from LLaVA's free-form text response."""
        content_lower = content.lower()
        
        # Extract visual description (first substantial paragraph or sentence)
        visual_desc = content[:500] if len(content) > 50 else content
        # Clean up
        visual_desc = visual_desc.replace('\n', ' ').strip()
        if len(visual_desc) > 300:
            visual_desc = visual_desc[:300] + "..."
        
        # Try to extract OCR text - look for quoted text or "text:" patterns
        ocr_text = ""
        text_patterns = [
            r'"([^"]{3,})"',  # Quoted text
            r'text[:\s]+["\']?([^"\'\n]{3,})["\']?',  # "text: ..."
            r'says?\s+["\']([^"\']+)["\']',  # says "..."
            r'reads?\s+["\']([^"\']+)["\']',  # reads "..."
        ]
        for pattern in text_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                ocr_text = matches[0]
                break
        
        # Determine hate risk level from content
        hate_risk = "LOW"
        high_indicators = ['hate', 'racist', 'offensive', 'discriminat', 'slur', 'derogatory', 
                          'attack', 'violent', 'harmful', 'toxic', 'threatening']
        medium_indicators = ['mock', 'stereotyp', 'inappropriate', 'sensitive', 'controversial',
                            'insult', 'degrad', 'demean', 'hurtful', 'offensive']
        
        for indicator in high_indicators:
            if indicator in content_lower:
                hate_risk = "HIGH"
                break
        
        if hate_risk != "HIGH":
            for indicator in medium_indicators:
                if indicator in content_lower:
                    hate_risk = "MEDIUM"
                    break
        
        # Extract target group if mentioned
        target_group = None
        group_patterns = [
            r'target(?:s|ing|ed)?\s+([a-zA-Z\s]+)',
            r'(?:against|toward)\s+([a-zA-Z\s]+)',
            r'([a-zA-Z]+)\s+(?:community|people|group)',
        ]
        for pattern in group_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                target_group = matches[0].strip()[:50]
                break
        
        return {
            "visual_description": visual_desc,
            "ocr_text": ocr_text,
            "implicit_meaning": content[:200] if len(content) > 200 else content,
            "target_group": target_group,
            "hate_risk_level": hate_risk
        }
    
    async def analyze_image(self, image_path: str) -> VLMOutput:
        """Analyze a meme image using local Ollama LLaVA."""
        image_data = self._encode_image(image_path)
        
        # Ollama API format for vision models
        payload = {
            "model": self.model,
            "prompt": self._get_prompt(),
            "images": [image_data],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1500
            }
        }
        
        async with httpx.AsyncClient(timeout=600) as client:  # 10 min timeout for CPU
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error {response.status_code}: {response.text}"
                )
            
            result = response.json()
        
        content = result.get("response", "")
        
        # Parse JSON response
        parsed = self._parse_json_response(content)
        
        return VLMOutput(
            visual_description=parsed.get("visual_description", ""),
            ocr_text=parsed.get("ocr_text", ""),
            implicit_meaning=parsed.get("implicit_meaning", ""),
            target_group=parsed.get("target_group"),
            hate_risk_level=parsed.get("hate_risk_level")
        )
    
    async def health_check(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    def set_prompt_mode(self, mode: Literal["standard", "quick", "code_mixed"]) -> None:
        """Change the prompt mode."""
        if mode not in self._prompts:
            raise ValueError(f"Invalid prompt mode: {mode}")
        self.prompt_mode = mode


class OllamaLLM:
    """
    Local LLM implementation using Ollama.
    """
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 500
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama LLM error: {response.text}")
            
            return response.json().get("response", "")
    
    async def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
