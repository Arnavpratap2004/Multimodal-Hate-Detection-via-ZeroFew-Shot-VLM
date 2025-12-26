"""
LLM (Large Language Model) module for hate detection reasoning.
"""

from .base import BaseLLM, ClassificationResult
from .base import BaseLLM, ClassificationResult
from .openrouter_llm import OpenRouterLLM
from .ollama_llm import OllamaLLM
from .zero_shot import ZeroShotClassifier
from .few_shot import FewShotClassifier
from .chain_of_thought import ChainOfThoughtClassifier

__all__ = [
    "BaseLLM",
    "ClassificationResult",
    "OpenRouterLLM",
    "OllamaLLM",
    "ZeroShotClassifier",
    "FewShotClassifier", 
    "ChainOfThoughtClassifier"
]
