"""
Main hate detection pipeline orchestrator.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Literal, Optional, Union

from rich.console import Console
from rich.progress import Progress, TaskID
from tqdm import tqdm

from ..config import settings
from ..vlm import OpenRouterVLM, VLMOutput
from ..llm import (
    OpenRouterLLM,
    ZeroShotClassifier,
    FewShotClassifier,
    ChainOfThoughtClassifier,
    ClassificationResult
)
from .schemas import FullAnalysis, BatchResult, DatasetSample


console = Console()


class HateDetector:
    """
    Main pipeline orchestrator for multimodal hate detection.
    
    Implements the strict pipeline:
    Image → VLM → Image description + OCR → LLM → HATE/NON-HATE
    """
    
    def __init__(
        self,
        vlm_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        vlm_prompt_mode: Literal["standard", "quick", "code_mixed"] = "standard"
    ):
        """
        Initialize the hate detector pipeline.
        
        Args:
            vlm_model: VLM model to use. Uses config default if None.
            llm_model: LLM model to use. Uses config default if None.
            api_key: API key for both VLM and LLM. Uses config default if None.
            vlm_prompt_mode: Prompt mode for VLM analysis.
        """
        # Initialize VLM
        self.vlm = OpenRouterVLM(
            model=vlm_model,
            api_key=api_key,
            prompt_mode=vlm_prompt_mode
        )
        
        # Initialize LLM
        self.llm = OpenRouterLLM(
            model=llm_model,
            api_key=api_key
        )
        
        # Initialize classifiers
        self.classifiers = {
            "zero_shot": ZeroShotClassifier(self.llm),
            "few_shot": FewShotClassifier(self.llm),
            "cot": ChainOfThoughtClassifier(OpenRouterLLM(
                model=llm_model,
                api_key=api_key,
                temperature=0.2
            ))
        }
    
    def _get_classifier(self, mode: str):
        """Get the classifier for the specified mode."""
        if mode not in self.classifiers:
            raise ValueError(f"Unknown inference mode: {mode}. Available: {list(self.classifiers.keys())}")
        return self.classifiers[mode]
    
    async def detect(
        self,
        image_path: Union[str, Path],
        mode: Literal["zero_shot", "few_shot", "cot"] = "zero_shot"
    ) -> FullAnalysis:
        """
        Detect hate speech in a meme image.
        
        Follows the strict pipeline:
        1. VLM: Image → Visual description + OCR + Cultural context
        2. LLM: Text analysis → HATE/NON-HATE classification
        
        Args:
            image_path: Path to the meme image.
            mode: Inference mode to use.
            
        Returns:
            FullAnalysis containing VLM output, classification, and metadata.
        """
        image_path = str(image_path)
        start_time = time.time()
        vlm_time = None
        llm_time = None
        error = None
        
        try:
            # Step 1: VLM Analysis
            vlm_start = time.time()
            vlm_output = await self.vlm.analyze_image(image_path)
            vlm_time = time.time() - vlm_start
            
            # Step 2: LLM Classification
            llm_start = time.time()
            classifier = self._get_classifier(mode)
            classification = await classifier.classify(vlm_output)
            llm_time = time.time() - llm_start
            
        except Exception as e:
            error = str(e)
            # Create placeholder outputs for failed analysis
            vlm_output = VLMOutput(
                visual_description="Analysis failed",
                ocr_text="",
                implicit_meaning="",
                target_group=None
            )
            classification = ClassificationResult(
                label="NON-HATE",
                justification=f"Error during analysis: {error}"
            )
        
        processing_time = time.time() - start_time
        
        return FullAnalysis(
            image_path=image_path,
            vlm_output=vlm_output,
            classification=classification,
            inference_mode=mode,
            processing_time=processing_time,
            vlm_time=vlm_time,
            llm_time=llm_time,
            error=error
        )
    
    async def detect_batch(
        self,
        image_paths: List[Union[str, Path]],
        mode: Literal["zero_shot", "few_shot", "cot"] = "zero_shot",
        show_progress: bool = True,
        max_concurrent: int = 5
    ) -> BatchResult:
        """
        Detect hate speech in multiple meme images.
        
        Args:
            image_paths: List of paths to meme images.
            mode: Inference mode to use for all images.
            show_progress: Whether to show a progress bar.
            max_concurrent: Maximum concurrent API calls.
            
        Returns:
            BatchResult containing all individual results and summary stats.
        """
        batch_start = time.time()
        batch_result = BatchResult(inference_mode=mode)
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(path: Union[str, Path]) -> FullAnalysis:
            async with semaphore:
                return await self.detect(path, mode)
        
        if show_progress:
            # Process with progress bar
            tasks = [process_with_semaphore(path) for path in image_paths]
            
            with Progress() as progress:
                task_id = progress.add_task(
                    f"[cyan]Processing ({mode})...",
                    total=len(image_paths)
                )
                
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    batch_result.add_result(result)
                    progress.advance(task_id)
        else:
            # Process without progress bar
            tasks = [process_with_semaphore(path) for path in image_paths]
            results = await asyncio.gather(*tasks)
            for result in results:
                batch_result.add_result(result)
        
        batch_result.total_time = time.time() - batch_start
        return batch_result
    
    async def compare_modes(
        self,
        image_path: Union[str, Path]
    ) -> dict[str, FullAnalysis]:
        """
        Run all three inference modes on a single image for comparison.
        
        Args:
            image_path: Path to the meme image.
            
        Returns:
            Dictionary mapping mode names to their results.
        """
        results = {}
        
        for mode in ["zero_shot", "few_shot", "cot"]:
            results[mode] = await self.detect(image_path, mode)
        
        return results
    
    async def health_check(self) -> dict[str, bool]:
        """
        Check the health of VLM and LLM services.
        
        Returns:
            Dictionary with health status of each service.
        """
        vlm_healthy = await self.vlm.health_check()
        llm_healthy = await self.llm.health_check()
        
        return {
            "vlm": vlm_healthy,
            "llm": llm_healthy,
            "overall": vlm_healthy and llm_healthy
        }
    
    def set_vlm_prompt_mode(
        self,
        mode: Literal["standard", "quick", "code_mixed"]
    ) -> None:
        """
        Change the VLM prompt mode.
        
        Args:
            mode: The prompt mode to use for VLM analysis.
        """
        self.vlm.set_prompt_mode(mode)


async def main():
    """Demo usage of the HateDetector."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.detector <image_path> [mode]")
        print("Modes: zero_shot, few_shot, cot")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "zero_shot"
    
    detector = HateDetector()
    
    console.print(f"\n[bold cyan]🔍 Analyzing meme: {image_path}[/]")
    console.print(f"[dim]Mode: {mode}[/]\n")
    
    result = await detector.detect(image_path, mode)
    
    if result.error:
        console.print(f"[bold red]❌ Error: {result.error}[/]")
    else:
        # Display VLM output
        console.print("[bold yellow]📸 VLM Analysis:[/]")
        console.print(f"  Visual: {result.vlm_output.visual_description[:200]}...")
        console.print(f"  OCR: {result.vlm_output.ocr_text}")
        console.print(f"  Meaning: {result.vlm_output.implicit_meaning[:200]}...")
        console.print(f"  Target: {result.vlm_output.target_group}")
        
        console.print()
        
        # Display classification
        label_color = "red" if result.classification.label == "HATE" else "green"
        console.print(f"[bold {label_color}]🏷️  Label: {result.classification.label}[/]")
        console.print(f"[dim]Justification: {result.classification.justification}[/]")
        console.print(f"[dim]Processing time: {result.processing_time:.2f}s[/]")


if __name__ == "__main__":
    asyncio.run(main())
