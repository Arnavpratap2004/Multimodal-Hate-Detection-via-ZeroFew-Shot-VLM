"""
Abstract base class for LLM reasoning and classification.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional
from pydantic import BaseModel, Field

from ..vlm.base import VLMOutput


class ClassificationResult(BaseModel):
    """
    Result of hate speech classification.
    
    This represents the final output of the pipeline.
    """
    
    label: Literal["HATE", "NON-HATE", "ERROR"] = Field(
        ...,
        description="Binary classification label (ERROR for failed analyses)"
    )
    
    justification: str = Field(
        ...,
        description="One-sentence explanation for the classification"
    )
    
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score (0-1)"
    )
    
    confidence_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = Field(
        default=None,
        description="Confidence level from LLM (LOW/MEDIUM/HIGH)"
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "label": self.label,
            "justification": self.justification
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.confidence_level is not None:
            result["confidence_level"] = self.confidence_level
        return result


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations.
    """
    
    @abstractmethod
    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt to the LLM and get a completion.
        
        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt for context.
            
        Returns:
            The LLM's response text.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if the service is healthy.
        """
        pass


class BaseClassifier(ABC):
    """
    Abstract base class for hate speech classifiers.
    
    All classifier implementations (zero-shot, few-shot, CoT) must
    inherit from this class.
    """
    
    def __init__(self, llm: BaseLLM):
        """
        Initialize the classifier with an LLM instance.
        
        Args:
            llm: The LLM to use for reasoning.
        """
        self.llm = llm
    
    @abstractmethod
    async def classify(self, vlm_output: VLMOutput) -> ClassificationResult:
        """
        Classify the meme content as HATE or NON-HATE.
        
        Args:
            vlm_output: Structured output from VLM image analysis.
            
        Returns:
            ClassificationResult with label and justification.
        """
        pass
    
    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Return the name of this classification mode."""
        pass
