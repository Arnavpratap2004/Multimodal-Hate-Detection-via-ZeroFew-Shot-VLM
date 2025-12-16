"""
Abstract base class for Vision-Language Models.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field


class VLMOutput(BaseModel):
    """
    Structured output from VLM image analysis.
    
    This represents the first stage of the pipeline where the VLM
    extracts visual and textual information from the meme image.
    """
    
    visual_description: str = Field(
        ...,
        description="Detailed description of the visual scene in the image"
    )
    
    ocr_text: str = Field(
        ...,
        description="All visible text extracted from the image using OCR"
    )
    
    implicit_meaning: str = Field(
        ...,
        description="Explanation of sarcasm, mockery, stereotypes, cultural/social/political references"
    )
    
    target_group: Optional[str] = Field(
        default=None,
        description="The target group being referenced or attacked (if any)"
    )
    
    def to_context_string(self) -> str:
        """
        Convert VLM output to a formatted string for LLM input.
        
        Returns:
            Formatted string containing all VLM analysis results.
        """
        parts = [
            f"**Visual Description:**\n{self.visual_description}",
            f"\n**Extracted Text (OCR):**\n{self.ocr_text}",
            f"\n**Implicit Meaning & Cultural Context:**\n{self.implicit_meaning}",
        ]
        
        if self.target_group:
            parts.append(f"\n**Identified Target Group:**\n{self.target_group}")
        else:
            parts.append("\n**Identified Target Group:**\nNone identified")
        
        return "\n".join(parts)


class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Model implementations.
    
    All VLM implementations must inherit from this class and implement
    the analyze_image method.
    """
    
    @abstractmethod
    async def analyze_image(self, image_path: str) -> VLMOutput:
        """
        Analyze a meme image and extract structured information.
        
        This method performs the first stage of the hate detection pipeline:
        1. Describe the visual scene in detail
        2. Extract all visible text using OCR
        3. Explain cultural, social, or political references
        4. Identify sarcasm, mockery, stereotypes, or symbolism
        5. Explicitly state the target group (if any)
        
        Args:
            image_path: Path to the meme image file.
            
        Returns:
            VLMOutput containing structured analysis results.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image format is not supported.
            RuntimeError: If the VLM API call fails.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the VLM service is available and configured correctly.
        
        Returns:
            True if the service is healthy, False otherwise.
        """
        pass
