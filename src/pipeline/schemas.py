"""
Pydantic schemas for the pipeline.
"""

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from ..vlm.base import VLMOutput
from ..llm.base import ClassificationResult


class FullAnalysis(BaseModel):
    """
    Complete analysis result for a single meme.
    
    Combines VLM output, classification result, and metadata.
    """
    
    image_path: str = Field(
        ...,
        description="Path to the analyzed meme image"
    )
    
    vlm_output: VLMOutput = Field(
        ...,
        description="Structured output from VLM analysis"
    )
    
    classification: ClassificationResult = Field(
        ...,
        description="Final classification result"
    )
    
    inference_mode: Literal["zero_shot", "few_shot", "cot"] = Field(
        ...,
        description="The inference mode used for classification"
    )
    
    processing_time: float = Field(
        ...,
        ge=0,
        description="Total processing time in seconds"
    )
    
    vlm_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="VLM processing time in seconds"
    )
    
    llm_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="LLM processing time in seconds"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the analysis was performed"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if analysis failed"
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_path": self.image_path,
            "vlm_output": self.vlm_output.model_dump(),
            "classification": self.classification.to_dict(),
            "inference_mode": self.inference_mode,
            "processing_time": self.processing_time,
            "vlm_time": self.vlm_time,
            "llm_time": self.llm_time,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Image: {self.image_path}",
            f"Label: {self.classification.label}",
            f"Justification: {self.classification.justification}",
            f"Mode: {self.inference_mode}",
            f"Processing Time: {self.processing_time:.2f}s"
        ]
        return "\n".join(lines)


class BatchResult(BaseModel):
    """
    Results from processing a batch of memes.
    """
    
    results: List[FullAnalysis] = Field(
        default_factory=list,
        description="List of individual analysis results"
    )
    
    total_count: int = Field(
        default=0,
        description="Total number of memes processed"
    )
    
    hate_count: int = Field(
        default=0,
        description="Number of memes classified as HATE"
    )
    
    non_hate_count: int = Field(
        default=0,
        description="Number of memes classified as NON-HATE"
    )
    
    error_count: int = Field(
        default=0,
        description="Number of failed analyses"
    )
    
    total_time: float = Field(
        default=0.0,
        description="Total batch processing time in seconds"
    )
    
    inference_mode: str = Field(
        default="",
        description="Inference mode used for the batch"
    )
    
    def add_result(self, result: FullAnalysis) -> None:
        """Add a result and update counts."""
        self.results.append(result)
        self.total_count += 1
        
        if result.error:
            self.error_count += 1
        elif result.classification.label == "HATE":
            self.hate_count += 1
        else:
            self.non_hate_count += 1
    
    @property
    def accuracy_summary(self) -> str:
        """Generate accuracy summary string."""
        if self.total_count == 0:
            return "No results to summarize"
        
        successful = self.total_count - self.error_count
        return (
            f"Total: {self.total_count} | "
            f"HATE: {self.hate_count} | "
            f"NON-HATE: {self.non_hate_count} | "
            f"Errors: {self.error_count} | "
            f"Success Rate: {successful/self.total_count*100:.1f}%"
        )


class DatasetSample(BaseModel):
    """
    A sample from a hate speech dataset.
    """
    
    id: str = Field(
        ...,
        description="Unique identifier for the sample"
    )
    
    image_path: str = Field(
        ...,
        description="Path to the meme image"
    )
    
    ground_truth_label: Literal["HATE", "NON-HATE"] = Field(
        ...,
        description="Ground truth label from the dataset"
    )
    
    text: Optional[str] = Field(
        default=None,
        description="Associated text if provided by dataset"
    )
    
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata from the dataset"
    )


class EvaluationResult(BaseModel):
    """
    Evaluation metrics for a dataset run.
    """
    
    dataset_name: str
    inference_mode: str
    total_samples: int
    
    # Core metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Confusion matrix
    true_positives: int  # Correctly identified HATE
    true_negatives: int  # Correctly identified NON-HATE
    false_positives: int  # Incorrectly labeled HATE
    false_negatives: int  # Incorrectly labeled NON-HATE
    
    # Additional info
    avg_processing_time: float
    error_rate: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
