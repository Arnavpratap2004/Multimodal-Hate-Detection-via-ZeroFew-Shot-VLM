"""
Bangla meme dataset loader.

Supports multiple Bangla hate speech datasets:
- BHM (Bengali Hateful Memes) - 7,148 memes
- MUTE - 4,158 memes
- BanglaAbuseMeme - 4,043 memes
"""

import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

from .base import BaseDatasetLoader, DatasetSample


class BanglaLoader(BaseDatasetLoader):
    """
    Loader for Bangla meme hate speech datasets.
    
    Expected directory structure:
    data_dir/
    ├── images/
    │   ├── 1.jpg
    │   └── ...
    └── annotations.json  (or annotations.csv)
    """
    
    # Supported dataset types
    DATASET_TYPES = ["bhm", "mute", "banglaabuse", "generic"]
    
    def __init__(
        self,
        data_dir: str | Path,
        dataset_type: Literal["bhm", "mute", "banglaabuse", "generic"] = "generic",
        annotation_file: str = "annotations.json"
    ):
        """
        Initialize the Bangla dataset loader.
        
        Args:
            data_dir: Path to the dataset directory.
            dataset_type: Type of Bangla dataset.
            annotation_file: Name of the annotation file.
        """
        super().__init__(data_dir)
        self.dataset_type = dataset_type
        self.annotation_file = annotation_file
        self.images_dir = self.data_dir / "images"
    
    @property
    def name(self) -> str:
        names = {
            "bhm": "Bengali Hateful Memes (BHM)",
            "mute": "MUTE",
            "banglaabuse": "BanglaAbuseMeme",
            "generic": "Bangla Meme Dataset"
        }
        return names.get(self.dataset_type, "Bangla Meme Dataset")
    
    @property
    def description(self) -> str:
        descriptions = {
            "bhm": "Bengali/code-mixed hateful meme detection with target entity identification",
            "mute": "Multimodal hateful meme detection for Bengali",
            "banglaabuse": "Bengali abusive meme classification with vulgarity and sarcasm labels",
            "generic": "Generic Bangla meme hate speech dataset"
        }
        return descriptions.get(self.dataset_type, "Bangla meme hate speech dataset")
    
    def _map_label(self, raw_label: Any) -> str:
        """
        Map dataset-specific labels to HATE/NON-HATE.
        
        Handles various label formats across Bangla datasets:
        - "hateful"/"non-hateful"
        - "abusive"/"non-abusive"
        - 1/0
        - "hate"/"not_hate"
        """
        if isinstance(raw_label, str):
            raw_label = raw_label.lower().strip()
            
            hate_keywords = [
                "hateful", "hate", "abusive", "abuse", "offensive",
                "yes", "1", "true", "positive"
            ]
            
            for keyword in hate_keywords:
                if keyword in raw_label:
                    return "HATE"
            
            return "NON-HATE"
        
        if isinstance(raw_label, (int, float)):
            return "HATE" if raw_label == 1 else "NON-HATE"
        
        if isinstance(raw_label, bool):
            return "HATE" if raw_label else "NON-HATE"
        
        return "NON-HATE"
    
    def _load_from_json(self) -> List[DatasetSample]:
        """Load annotations from JSON file."""
        annotation_path = self.data_dir / self.annotation_file
        
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        samples = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "data" in data:
                items = data["data"]
            elif "samples" in data:
                items = data["samples"]
            else:
                items = list(data.values())
        else:
            items = []
        
        for item in items:
            sample_id = str(item.get("id", item.get("image_id", item.get("idx", len(samples)))))
            
            # Find image path
            image_name = item.get("image", item.get("image_path", item.get("filename", item.get("img"))))
            if image_name:
                image_path = self.images_dir / image_name
            else:
                image_path = self.images_dir / f"{sample_id}.jpg"
            
            # Get label - try multiple possible keys
            raw_label = None
            label_keys = ["label", "hate", "hateful", "abusive", "is_hateful", "class"]
            for key in label_keys:
                if key in item:
                    raw_label = item[key]
                    break
            
            label = self._map_label(raw_label)
            
            # Get text/caption
            text = item.get("text", item.get("caption", item.get("ocr_text", item.get("meme_text"))))
            
            # Extract metadata
            metadata_keys = [
                "target", "target_entity", "sentiment", "sarcasm", 
                "vulgarity", "emotion", "confidence"
            ]
            metadata = {k: item[k] for k in metadata_keys if k in item}
            
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
                
                # Find label
                raw_label = None
                for key in ["label", "hate", "hateful", "abusive"]:
                    if key in row and row[key]:
                        raw_label = row[key]
                        break
                
                label = self._map_label(raw_label)
                
                text = row.get("text", row.get("caption", row.get("ocr_text")))
                
                # Collect metadata
                metadata = {}
                for key in ["target", "sarcasm", "vulgarity", "sentiment"]:
                    if key in row and row[key]:
                        metadata[key] = row[key]
                
                samples.append(DatasetSample(
                    id=sample_id,
                    image_path=str(image_path),
                    ground_truth_label=label,
                    text=text,
                    metadata=metadata if metadata else None
                ))
        
        return samples
    
    def load(self) -> List[DatasetSample]:
        """
        Load the Bangla meme dataset.
        
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
        """Get dataset statistics including Bangla-specific info."""
        stats = super().get_statistics()
        
        if self._samples:
            # Count by additional labels if available
            with_target = sum(
                1 for s in self._samples
                if s.metadata and s.metadata.get("target")
            )
            
            with_sarcasm = sum(
                1 for s in self._samples
                if s.metadata and s.metadata.get("sarcasm")
            )
            
            stats["with_target_count"] = with_target
            stats["with_sarcasm_count"] = with_sarcasm
            stats["has_text_count"] = sum(1 for s in self._samples if s.text)
        
        return stats
