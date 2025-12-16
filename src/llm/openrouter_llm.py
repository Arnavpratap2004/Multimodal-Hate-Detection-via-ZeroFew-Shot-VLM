"""
OpenRouter-based LLM implementation.
"""

import json
import os
import re
from typing import Optional

import httpx

from ..config import settings
from .base import BaseLLM


class OpenRouterLLM(BaseLLM):
    """
    LLM implementation using OpenRouter API.
    
    Supports multiple LLM backends including GPT-4, Claude, and Llama.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Initialize the OpenRouter LLM client.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o"). Uses config default if None.
            api_key: OpenRouter API key. Uses config default if None.
            temperature: Sampling temperature (0-1). Lower = more deterministic.
        """
        self.model = model or settings.llm_model
        # Try multiple sources for API key: parameter > env var > settings
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY in your .env file or pass api_key parameter."
            )
    
    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt to the LLM and get a completion.
        
        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt for context.
            
        Returns:
            The LLM's response text.
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
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
                    "temperature": self.temperature,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code != 200:
                error_detail = response.text
                raise RuntimeError(
                    f"LLM API request failed with status {response.status_code}: {error_detail}"
                )
            
            result = response.json()
        
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected API response format: {result}") from e
    
    async def health_check(self) -> bool:
        """
        Check if the OpenRouter API is accessible.
        
        Returns:
            True if the API is healthy.
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
    
    @staticmethod
    def parse_json_from_response(content: str) -> dict:
        """
        Parse JSON from LLM response, handling various formats.
        
        Args:
            content: Raw response content from LLM.
            
        Returns:
            Parsed JSON dictionary.
        """
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to extract from markdown code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON object
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        
        raise ValueError(f"Could not parse JSON from response: {content[:300]}...")
