"""
Pipeline module for orchestrating the hate detection workflow.
"""

from .detector import HateDetector
from .schemas import FullAnalysis, BatchResult

__all__ = ["HateDetector", "FullAnalysis", "BatchResult"]
