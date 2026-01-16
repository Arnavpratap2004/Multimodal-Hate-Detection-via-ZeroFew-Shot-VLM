"""
VLM (Vision-Language Model) module for image understanding.
"""

from .base import BaseVLM, VLMOutput
from .openrouter_vlm import OpenRouterVLM

__all__ = ["BaseVLM", "VLMOutput", "OpenRouterVLM"]
