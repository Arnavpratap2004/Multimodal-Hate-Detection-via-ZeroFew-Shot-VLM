"""
Evaluation module for hate detection metrics and analysis.
"""

from .metrics import calculate_metrics, MetricsCalculator
from .analyzer import FailureModeAnalyzer

__all__ = ["calculate_metrics", "MetricsCalculator", "FailureModeAnalyzer"]
