"""
Ollama-based local LLM implementation.
"""

from typing import List, Optional
import httpx

from ..config import settings
from .base import BaseLLM, ClassificationResult


class OllamaLLM(BaseLLM):
    """
    LLM implementation using local Ollama server.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Initialize the Ollama LLM client.
        
        Args:
            model: Model identifier (e.g., "llama3"). Uses config default if None.
            base_url: Ollama API base URL. Uses config default if None.
            temperature: Sampling temperature.
        """
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature
        
    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: User prompt.
            system_prompt: Optional system instruction.
            
        Returns:
            Generated text response.
        """
        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Make API request - use long timeout for CPU inference
        async with httpx.AsyncClient(timeout=600) as client:  # 10 min for CPU
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 500  # Reasonable limit for classification
                    }
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API request failed with status {response.status_code}: {response.text}"
                )
            
            result = response.json()
            
        return result.get("message", {}).get("content", "")

    async def health_check(self) -> bool:
        """
        Check if the Ollama API is accessible.
        
        Returns:
            True if the API is healthy.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
