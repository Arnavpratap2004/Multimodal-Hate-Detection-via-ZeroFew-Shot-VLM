"""
Configuration management for the Multimodal Hate Detection system.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

# Try to import pydantic-settings, fall back to BaseModel if not available
try:
    from pydantic_settings import BaseSettings
except ImportError:
    BaseSettings = BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_provider: Literal["openrouter", "ollama"] = Field(
        default="openrouter",
        description="API provider to use (openrouter or ollama)"
    )
    
    # OpenRouter Settings
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key for VLM and LLM access"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    
    # Ollama Settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    
    # Model Selection
    vlm_model: str = Field(
        default="meta-llama/llama-3.2-11b-vision-instruct:free",
        description="Vision-Language Model to use"
    )
    llm_model: str = Field(
        default="qwen/qwen-2.5-72b-instruct:free",
        description="Large Language Model for reasoning"
    )
    
    # Inference Settings
    default_inference_mode: Literal["zero_shot", "few_shot", "cot"] = Field(
        default="zero_shot",
        description="Default inference mode"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum API retry attempts"
    )
    request_timeout: int = Field(
        default=60,
        description="API request timeout in seconds"
    )
    
    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory"
    )
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self.project_root / "data"
    
    @property
    def samples_dir(self) -> Path:
        """Get the samples directory path."""
        return self.data_dir / "samples"
    
    @property
    def datasets_dir(self) -> Path:
        """Get the datasets directory path."""
        return self.data_dir / "datasets"
    
    @property
    def results_dir(self) -> Path:
        """Get the results directory path."""
        return self.project_root / "results"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


# Available VLM models
VLM_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-pro-1.5",
    "google/gemini-flash-1.5",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
]

# Available LLM models
LLM_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]

# Inference modes
INFERENCE_MODES = ["zero_shot", "few_shot", "cot"]
