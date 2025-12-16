"""
MultiBully dataset loader.

MultiBully (SIGIR 2022) - Code-mixed Hindi-English cyberbullying memes
- 5,854 memes (3,222 bully, 2,632 non-bully)
- Labels: cyberbullying, sentiment, emotion, sarcasm

Dataset: https://github.com/hate-alert/Multibully
"""

import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import BaseDatasetLoader, DatasetSample


class MultiBullyLoader(BaseDatasetLoader):
    """
    Loader for the MultiBully dataset.
    
    Expected directory structure:
    data_dir/
    ├── images/
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── annotations.json  (or annotations.csv)
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        annotation_file: str = "annotations.json"
    ):
        """
        Initialize the MultiBully loader.
        
        Args:
            data_dir: Path to the MultiBully dataset directory.
            annotation_file: Name of the annotation file.
        """
        super().__init__(data_dir)
        self.annotation_file = annotation_file
        self.images_dir = self.data_dir / "images"
    
    @property
    def name(self) -> str:
        return "MultiBully"
    
    @property
    def description(self) -> str:
        return "Hindi-English code-mixed cyberbullying memes (SIGIR 2022)"
    
    def _map_label(self, raw_label: Any) -> str:
        """
        Map dataset-specific labels to HATE/NON-HATE.
        
        MultiBully uses:
        - 1 or "bully" -> HATE
        - 0 or "non-bully" -> NON-HATE
        """
        if isinstance(raw_label, str):
            raw_label = raw_label.lower()
            if raw_label in ["bully", "bullying", "yes", "1", "true"]:
                return "HATE"
            return "NON-HATE"
        
        if isinstance(raw_label, (int, float)):
            return "HATE" if raw_label == 1 else "NON-HATE"
        
        return "NON-HATE"  # Default
    
    def _load_from_json(self) -> List[DatasetSample]:
        """Load annotations from JSON file."""
        annotation_path = self.data_dir / self.annotation_file
        
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "data" in data:
            items = data["data"]
        else:
            items = list(data.values()) if isinstance(data, dict) else []
        
        for item in items:
            # Try different key names
            sample_id = str(item.get("id", item.get("image_id", len(samples))))
            
            # Find image path
            image_name = item.get("image", item.get("image_path", item.get("filename")))
            if image_name:
                image_path = self.images_dir / image_name
            else:
                image_path = self.images_dir / f"{sample_id}.jpg"
            
            # Get label
            raw_label = item.get("label", item.get("bully", item.get("cyberbullying")))
            label = self._map_label(raw_label)
            
            # Get text if available
            text = item.get("text", item.get("ocr_text", None))
            
            # Additional metadata
            metadata = {
                k: v for k, v in item.items()
                if k not in ["id", "image_id", "image", "image_path", "filename", "label", "bully", "text"]
            }
            
            samples.append(DatasetSample(
                id=sample_id,
                image_path=str(image_path),
                ground_truth_label=label,
                text=text,
                metadata=metadata if metadata else None
            ))
        
        return samples
    
    def _load_from_csv(self) -> List[DatasetSample]:
        """Load annotations from CSV file."""
        annotation_path = self.data_dir / self.annotation_file.replace(".json", ".csv")
        
        samples = []
        
        with open(annotation_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader):
                sample_id = row.get("id", row.get("image_id", str(idx)))
                
                image_name = row.get("image", row.get("image_path", row.get("filename")))
                if image_name:
                    image_path = self.images_dir / image_name
                else:
                    image_path = self.images_dir / f"{sample_id}.jpg"
                
                raw_label = row.get("label", row.get("bully", row.get("cyberbullying")))
                label = self._map_label(raw_label)
                
                text = row.get("text", row.get("ocr_text", None))
                
                samples.append(DatasetSample(
                    id=sample_id,
                    image_path=str(image_path),
                    ground_truth_label=label,
                    text=text,
                    metadata={k: v for k, v in row.items() if k not in ["id", "image", "label", "text"]}
                ))
        
        return samples
    
    def load(self) -> List[DatasetSample]:
        """
        Load the MultiBully dataset.
        
        Returns:
            List of DatasetSample objects.
        """
        annotation_path = self.data_dir / self.annotation_file
        
        if annotation_path.exists():
            if self.annotation_file.endswith(".json"):
                return self._load_from_json()
            elif self.annotation_file.endswith(".csv"):
                return self._load_from_csv()
        
        # Try CSV if JSON not found
        csv_path = self.data_dir / self.annotation_file.replace(".json", ".csv")
        if csv_path.exists():
            self.annotation_file = csv_path.name
            return self._load_from_csv()
        
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}\n"
            f"Expected structure:\n"
            f"  {self.data_dir}/\n"
            f"    images/\n"
            f"    annotations.json (or .csv)"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics including MultiBully-specific info."""
        stats = super().get_statistics()
        
        if self._samples:
            # Count by additional labels if available
            sarcasm_count = sum(
                1 for s in self._samples
                if s.metadata and s.metadata.get("sarcasm") == 1
            )
            
            stats["sarcasm_count"] = sarcasm_count
            stats["has_text_count"] = sum(1 for s in self._samples if s.text)
        
        return stats
