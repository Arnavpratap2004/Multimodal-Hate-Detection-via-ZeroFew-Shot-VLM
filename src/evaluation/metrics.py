"""
Metrics calculation for hate detection evaluation.
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from ..pipeline.schemas import FullAnalysis, EvaluationResult


@dataclass
class ConfusionMatrixData:
    """Confusion matrix breakdown."""
    true_positives: int   # Correctly predicted HATE
    true_negatives: int   # Correctly predicted NON-HATE
    false_positives: int  # Incorrectly predicted HATE (was NON-HATE)
    false_negatives: int  # Incorrectly predicted NON-HATE (was HATE)


def calculate_metrics(
    predictions: List[str],
    ground_truths: List[str],
    positive_label: str = "HATE"
) -> Dict[str, Any]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: List of predicted labels ("HATE" or "NON-HATE")
        ground_truths: List of ground truth labels
        positive_label: The label considered positive for precision/recall
        
    Returns:
        Dictionary containing all metrics.
    """
    # Convert to binary for sklearn
    y_pred = [1 if p == positive_label else 0 for p in predictions]
    y_true = [1 if g == positive_label else 0 for g in ground_truths]
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "total_samples": len(predictions),
        "positive_label": positive_label
    }


class MetricsCalculator:
    """
    Comprehensive metrics calculator for hate detection evaluation.
    """
    
    def __init__(self, positive_label: str = "HATE"):
        """
        Initialize the metrics calculator.
        
        Args:
            positive_label: The label considered positive for precision/recall.
        """
        self.positive_label = positive_label
        self.results: List[Tuple[str, str]] = []  # (prediction, ground_truth)
    
    def add_result(self, prediction: str, ground_truth: str) -> None:
        """Add a single result."""
        self.results.append((prediction, ground_truth))
    
    def add_results_from_analysis(
        self,
        analyses: List[FullAnalysis],
        ground_truths: List[str]
    ) -> None:
        """
        Add results from a list of FullAnalysis objects.
        
        Args:
            analyses: List of FullAnalysis objects from the detector.
            ground_truths: Corresponding ground truth labels.
        """
        for analysis, gt in zip(analyses, ground_truths):
            if not analysis.error:
                self.add_result(analysis.classification.label, gt)
    
    def calculate(self) -> Dict[str, Any]:
        """Calculate all metrics for accumulated results."""
        if not self.results:
            return {"error": "No results to calculate"}
        
        predictions = [r[0] for r in self.results]
        ground_truths = [r[1] for r in self.results]
        
        return calculate_metrics(predictions, ground_truths, self.positive_label)
    
    def get_evaluation_result(
        self,
        dataset_name: str,
        inference_mode: str,
        avg_processing_time: float = 0.0,
        error_rate: float = 0.0
    ) -> EvaluationResult:
        """
        Get a structured EvaluationResult object.
        
        Args:
            dataset_name: Name of the evaluated dataset.
            inference_mode: Inference mode used.
            avg_processing_time: Average processing time per sample.
            error_rate: Percentage of failed analyses.
            
        Returns:
            EvaluationResult object.
        """
        metrics = self.calculate()
        
        if "error" in metrics:
            # Return empty result if no data
            return EvaluationResult(
                dataset_name=dataset_name,
                inference_mode=inference_mode,
                total_samples=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                true_positives=0,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
                avg_processing_time=avg_processing_time,
                error_rate=error_rate
            )
        
        return EvaluationResult(
            dataset_name=dataset_name,
            inference_mode=inference_mode,
            total_samples=metrics["total_samples"],
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            true_positives=metrics["true_positives"],
            true_negatives=metrics["true_negatives"],
            false_positives=metrics["false_positives"],
            false_negatives=metrics["false_negatives"],
            avg_processing_time=avg_processing_time,
            error_rate=error_rate
        )
    
    def reset(self) -> None:
        """Clear all accumulated results."""
        self.results = []
    
    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get precision, recall, F1 for each class.
        
        Returns:
            Dictionary with metrics for each class.
        """
        if not self.results:
            return {}
        
        predictions = [r[0] for r in self.results]
        ground_truths = [r[1] for r in self.results]
        
        report = classification_report(
            ground_truths,
            predictions,
            labels=["HATE", "NON-HATE"],
            output_dict=True,
            zero_division=0
        )
        
        return {
            "HATE": {
                "precision": report.get("HATE", {}).get("precision", 0),
                "recall": report.get("HATE", {}).get("recall", 0),
                "f1_score": report.get("HATE", {}).get("f1-score", 0),
                "support": report.get("HATE", {}).get("support", 0)
            },
            "NON-HATE": {
                "precision": report.get("NON-HATE", {}).get("precision", 0),
                "recall": report.get("NON-HATE", {}).get("recall", 0),
                "f1_score": report.get("NON-HATE", {}).get("f1-score", 0),
                "support": report.get("NON-HATE", {}).get("support", 0)
            }
        }
    
    def print_report(self) -> str:
        """
        Generate a formatted text report of all metrics.
        
        Returns:
            Formatted string report.
        """
        metrics = self.calculate()
        
        if "error" in metrics:
            return "No results available for report."
        
        report = []
        report.append("=" * 50)
        report.append("HATE DETECTION EVALUATION REPORT")
        report.append("=" * 50)
        report.append(f"Total Samples: {metrics['total_samples']}")
        report.append("")
        report.append("OVERALL METRICS")
        report.append("-" * 30)
        report.append(f"Accuracy:  {metrics['accuracy']:.4f}")
        report.append(f"Precision: {metrics['precision']:.4f}")
        report.append(f"Recall:    {metrics['recall']:.4f}")
        report.append(f"F1 Score:  {metrics['f1_score']:.4f}")
        report.append("")
        report.append("CONFUSION MATRIX")
        report.append("-" * 30)
        report.append(f"True Positives (TP):  {metrics['true_positives']}")
        report.append(f"True Negatives (TN):  {metrics['true_negatives']}")
        report.append(f"False Positives (FP): {metrics['false_positives']}")
        report.append(f"False Negatives (FN): {metrics['false_negatives']}")
        report.append("")
        
        per_class = self.get_per_class_metrics()
        report.append("PER-CLASS METRICS")
        report.append("-" * 30)
        for label, class_metrics in per_class.items():
            report.append(f"{label}:")
            report.append(f"  Precision: {class_metrics['precision']:.4f}")
            report.append(f"  Recall:    {class_metrics['recall']:.4f}")
            report.append(f"  F1 Score:  {class_metrics['f1_score']:.4f}")
            report.append(f"  Support:   {class_metrics['support']}")
        
        report.append("=" * 50)
        
        return "\n".join(report)


def compare_modes(
    results_by_mode: Dict[str, List[Tuple[str, str]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare metrics across different inference modes.
    
    Args:
        results_by_mode: Dictionary mapping mode names to (prediction, ground_truth) tuples.
        
    Returns:
        Dictionary with metrics for each mode.
    """
    comparison = {}
    
    for mode, results in results_by_mode.items():
        calc = MetricsCalculator()
        for pred, gt in results:
            calc.add_result(pred, gt)
        comparison[mode] = calc.calculate()
    
    return comparison


def format_comparison_table(comparison: Dict[str, Dict[str, Any]]) -> str:
    """
    Format a comparison of modes as a table.
    
    Args:
        comparison: Dictionary from compare_modes().
        
    Returns:
        Formatted table string.
    """
    lines = []
    lines.append("| Mode | Accuracy | Precision | Recall | F1 Score |")
    lines.append("|------|----------|-----------|--------|----------|")
    
    for mode, metrics in comparison.items():
        if "error" not in metrics:
            lines.append(
                f"| {mode} | "
                f"{metrics['accuracy']:.4f} | "
                f"{metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | "
                f"{metrics['f1_score']:.4f} |"
            )
    
    return "\n".join(lines)
