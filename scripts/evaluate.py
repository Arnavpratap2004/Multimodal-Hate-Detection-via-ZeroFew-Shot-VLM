"""
Main evaluation script for the hate detection system.

Usage:
    python scripts/evaluate.py --dataset multibully --mode all
    python scripts/evaluate.py --dataset bangla --mode cot --n 100
    python scripts/evaluate.py --image path/to/meme.jpg --mode zero_shot
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID

from src.config import settings
from src.pipeline import HateDetector
from src.evaluation import MetricsCalculator, FailureModeAnalyzer
from data.datasets import MultiBullyLoader, BanglaLoader

console = Console()


async def evaluate_single_image(
    image_path: str,
    mode: str = "zero_shot",
    compare_all: bool = False
):
    """Evaluate a single meme image."""
    detector = HateDetector()
    
    console.print(f"\n[bold cyan]🔍 Analyzing: {image_path}[/]")
    
    if compare_all:
        console.print("[dim]Comparing all inference modes...[/]\n")
        results = await detector.compare_modes(image_path)
        
        table = Table(title="Mode Comparison")
        table.add_column("Mode", style="cyan")
        table.add_column("Label", style="bold")
        table.add_column("Justification")
        table.add_column("Time (s)")
        
        for mode_name, result in results.items():
            label_style = "red" if result.classification.label == "HATE" else "green"
            table.add_row(
                mode_name,
                f"[{label_style}]{result.classification.label}[/]",
                result.classification.justification[:50] + "...",
                f"{result.processing_time:.2f}"
            )
        
        console.print(table)
        
        # Show VLM output once
        console.print("\n[bold yellow]📸 VLM Analysis:[/]")
        vlm = results["zero_shot"].vlm_output
        console.print(f"  [dim]Visual:[/] {vlm.visual_description[:200]}...")
        console.print(f"  [dim]OCR:[/] {vlm.ocr_text}")
        console.print(f"  [dim]Meaning:[/] {vlm.implicit_meaning[:200]}...")
        console.print(f"  [dim]Target:[/] {vlm.target_group}")
        
    else:
        result = await detector.detect(image_path, mode)
        
        if result.error:
            console.print(f"[bold red]❌ Error: {result.error}[/]")
            return
        
        # Show VLM output
        console.print("\n[bold yellow]📸 VLM Analysis:[/]")
        console.print(f"  [dim]Visual:[/] {result.vlm_output.visual_description[:200]}...")
        console.print(f"  [dim]OCR:[/] {result.vlm_output.ocr_text}")
        console.print(f"  [dim]Meaning:[/] {result.vlm_output.implicit_meaning[:200]}...")
        console.print(f"  [dim]Target:[/] {result.vlm_output.target_group}")
        
        # Show classification
        label_color = "red" if result.classification.label == "HATE" else "green"
        console.print(f"\n[bold {label_color}]🏷️  Result: {result.classification.label}[/]")
        console.print(f"[dim]Justification: {result.classification.justification}[/]")
        console.print(f"[dim]Mode: {mode} | Time: {result.processing_time:.2f}s[/]")


async def evaluate_dataset(
    dataset_name: str,
    dataset_path: str,
    mode: str = "all",
    n_samples: int = None,
    output_dir: str = None
):
    """Evaluate on a dataset."""
    console.print(f"\n[bold cyan]📊 Dataset Evaluation: {dataset_name}[/]")
    
    # Load dataset
    if dataset_name.lower() == "multibully":
        loader = MultiBullyLoader(dataset_path)
    elif dataset_name.lower() in ["bangla", "bhm", "mute"]:
        loader = BanglaLoader(dataset_path)
    else:
        console.print(f"[red]Unknown dataset: {dataset_name}[/]")
        return
    
    try:
        samples = loader.get_samples(n=n_samples, shuffle=True)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        return
    
    console.print(f"Loaded {len(samples)} samples from {loader.name}")
    
    # Show dataset stats
    stats = loader.get_statistics()
    console.print(f"[dim]  HATE: {stats['hate_count']} | NON-HATE: {stats['non_hate_count']}[/]")
    
    # Determine modes to run
    modes = ["zero_shot", "few_shot", "cot"] if mode == "all" else [mode]
    
    # Initialize detector
    detector = HateDetector()
    
    # Results storage
    all_results = {}
    
    for current_mode in modes:
        console.print(f"\n[yellow]Running {current_mode} mode...[/]")
        
        calculator = MetricsCalculator()
        analyzer = FailureModeAnalyzer()
        
        total_time = 0
        error_count = 0
        
        with Progress() as progress:
            task = progress.add_task(f"[cyan]{current_mode}", total=len(samples))
            
            for sample in samples:
                try:
                    result = await detector.detect(sample.image_path, current_mode)
                    
                    if result.error:
                        error_count += 1
                    else:
                        calculator.add_result(
                            result.classification.label,
                            sample.ground_truth_label
                        )
                        
                        # Check for failures
                        if result.classification.label != sample.ground_truth_label:
                            analyzer.add_failure(result, sample.ground_truth_label)
                        
                        total_time += result.processing_time
                        
                except Exception as e:
                    error_count += 1
                    console.print(f"[red]Error processing {sample.id}: {e}[/]")
                
                progress.advance(task)
        
        # Calculate metrics
        metrics = calculator.calculate()
        successful = len(samples) - error_count
        
        # Store results
        all_results[current_mode] = {
            "metrics": metrics,
            "avg_time": total_time / successful if successful > 0 else 0,
            "error_rate": error_count / len(samples),
            "failure_report": analyzer.generate_report()
        }
        
        # Print metrics
        console.print(f"\n[bold green]Results for {current_mode}:[/]")
        console.print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        console.print(f"  Precision: {metrics.get('precision', 0):.4f}")
        console.print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        console.print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
        console.print(f"  Errors:    {error_count}/{len(samples)}")
    
    # Compare modes if multiple
    if len(modes) > 1:
        console.print("\n[bold cyan]📈 Mode Comparison:[/]")
        
        table = Table()
        table.add_column("Mode")
        table.add_column("Accuracy")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1")
        table.add_column("Avg Time")
        
        for mode_name, results in all_results.items():
            m = results["metrics"]
            table.add_row(
                mode_name,
                f"{m.get('accuracy', 0):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('f1_score', 0):.4f}",
                f"{results['avg_time']:.2f}s"
            )
        
        console.print(table)
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"eval_{dataset_name}_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "n_samples": len(samples),
                "timestamp": timestamp,
                "results": {
                    mode: {
                        "metrics": res["metrics"],
                        "avg_time": res["avg_time"],
                        "error_rate": res["error_rate"]
                    }
                    for mode, res in all_results.items()
                }
            }, f, indent=2)
        
        console.print(f"\n[dim]Results saved to: {result_file}[/]")


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the multimodal hate detection system"
    )
    
    # Single image evaluation
    parser.add_argument(
        "--image", "-i",
        help="Path to a single meme image to analyze"
    )
    
    # Dataset evaluation
    parser.add_argument(
        "--dataset", "-d",
        choices=["multibully", "bangla", "bhm", "mute"],
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset directory"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        default="zero_shot",
        choices=["zero_shot", "few_shot", "cot", "all"],
        help="Inference mode to use"
    )
    
    # Number of samples
    parser.add_argument(
        "--n", "-n",
        type=int,
        help="Number of samples to evaluate (for datasets)"
    )
    
    # Compare all modes for single image
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all modes on a single image"
    )
    
    # Output directory
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.dataset:
        parser.print_help()
        console.print("\n[red]Error: Specify either --image or --dataset[/]")
        return
    
    if args.image:
        await evaluate_single_image(
            args.image,
            args.mode if not args.compare else "zero_shot",
            compare_all=args.compare
        )
    elif args.dataset:
        if not args.dataset_path:
            args.dataset_path = settings.datasets_dir / args.dataset
        
        await evaluate_dataset(
            args.dataset,
            args.dataset_path,
            args.mode,
            args.n,
            args.output
        )


if __name__ == "__main__":
    asyncio.run(main())
