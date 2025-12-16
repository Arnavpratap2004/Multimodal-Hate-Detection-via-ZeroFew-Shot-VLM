"""
Base class for dataset loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any
from dataclasses import dataclass


@dataclass
class DatasetSample:
    """A single sample from a hate speech dataset."""
    id: str
    image_path: str
    ground_truth_label: str  # "HATE" or "NON-HATE"
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    All dataset loaders must implement the load() method to return
    an iterator of DatasetSamples.
    """
    
    def __init__(self, data_dir: str | Path):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Path to the dataset directory.
        """
        self.data_dir = Path(data_dir)
        self._samples: Optional[List[DatasetSample]] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the dataset description."""
        pass
    
    @abstractmethod
    def load(self) -> List[DatasetSample]:
        """
        Load the dataset and return all samples.
        
        Returns:
            List of DatasetSample objects.
        """
        pass
    
    def __iter__(self) -> Iterator[DatasetSample]:
        """Iterate over dataset samples."""
        if self._samples is None:
            self._samples = self.load()
        return iter(self._samples)
    
    def __len__(self) -> int:
        """Return number of samples."""
        if self._samples is None:
            self._samples = self.load()
        return len(self._samples)
    
    def get_sample(self, index: int) -> DatasetSample:
        """Get a specific sample by index."""
        if self._samples is None:
            self._samples = self.load()
        return self._samples[index]
    
    def get_samples(
        self,
        n: Optional[int] = None,
        shuffle: bool = False
    ) -> List[DatasetSample]:
        """
        Get samples from the dataset.
        
        Args:
            n: Number of samples to return. None for all.
            shuffle: Whether to shuffle before returning.
            
        Returns:
            List of DatasetSample objects.
        """
        if self._samples is None:
            self._samples = self.load()
        
        samples = self._samples.copy()
        
        if shuffle:
            import random
            random.shuffle(samples)
        
        if n is not None:
            samples = samples[:n]
        
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics.
        """
        if self._samples is None:
            self._samples = self.load()
        
        hate_count = sum(1 for s in self._samples if s.ground_truth_label == "HATE")
        non_hate_count = len(self._samples) - hate_count
        
        return {
            "dataset_name": self.name,
            "total_samples": len(self._samples),
            "hate_count": hate_count,
            "non_hate_count": non_hate_count,
            "hate_ratio": hate_count / len(self._samples) if self._samples else 0
        }
    
    def validate_paths(self) -> Dict[str, Any]:
        """
        Validate that all image paths exist.
        
        Returns:
            Dictionary with validation results.
        """
        if self._samples is None:
            self._samples = self.load()
        
        missing = []
        valid = []
        
        for sample in self._samples:
            if Path(sample.image_path).exists():
                valid.append(sample.id)
            else:
                missing.append(sample.id)
        
        return {
            "valid_count": len(valid),
            "missing_count": len(missing),
            "missing_ids": missing[:10] if missing else []  # First 10 missing
        }
