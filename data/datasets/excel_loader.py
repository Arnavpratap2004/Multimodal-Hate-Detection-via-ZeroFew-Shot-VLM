"""
Custom loader for the user's MultiBully Excel dataset.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DatasetSample:
    """A single sample from the dataset."""
    id: str
    image_path: str
    ground_truth_label: str  # "HATE" or "NON-HATE"
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MultiBullyExcelLoader:
    """
    Loader for MultiBully dataset with Excel annotations.
    
    Works with the Cyberbully_corrected_emotion_sentiment.xlsx file.
    """
    
    def __init__(
        self,
        images_dir: str,
        excel_path: str,
        label_column: str = "Img-Text-Label"
    ):
        """
        Initialize the loader.
        
        Args:
            images_dir: Path to the directory containing meme images.
            excel_path: Path to the Excel annotation file.
            label_column: Column name containing the hate label.
        """
        self.images_dir = Path(images_dir)
        self.excel_path = Path(excel_path)
        self.label_column = label_column
        self._samples: Optional[List[DatasetSample]] = None
        self._df: Optional[pd.DataFrame] = None
    
    @property
    def name(self) -> str:
        return "MultiBully (Excel)"
    
    def _load_excel(self) -> pd.DataFrame:
        """Load and cache the Excel file."""
        if self._df is None:
            self._df = pd.read_excel(self.excel_path)
        return self._df
    
    def _map_label(self, raw_label: Any) -> str:
        """Map Excel labels to HATE/NON-HATE."""
        if pd.isna(raw_label):
            return "NON-HATE"
        
        label = str(raw_label).lower().strip()
        
        # Check for non-hate labels FIRST (before checking for hate keywords)
        non_hate_keywords = [
            "nonbully", "non-bully", "non_bully", "not_bully",
            "harmless", "non-hate", "not_hate", "benign", "safe"
        ]
        for keyword in non_hate_keywords:
            if keyword in label.replace(" ", "").replace("-", "").replace("_", ""):
                return "NON-HATE"
        
        # Then check for hate-related labels
        hate_keywords = [
            "bully", "bullying", "harmful", "hateful", "hate", 
            "offensive", "abusive", "toxic"
        ]
        
        for keyword in hate_keywords:
            if keyword in label:
                return "HATE"
        
        # Check numeric values
        if label in ["1", "yes", "true"]:
            return "HATE"
        if label in ["0", "no", "false"]:
            return "NON-HATE"
        
        return "NON-HATE"  # Default
    
    def load(self) -> List[DatasetSample]:
        """Load all samples from the dataset."""
        if self._samples is not None:
            return self._samples
        
        df = self._load_excel()
        samples = []
        
        for idx, row in df.iterrows():
            # Get image name
            img_name = row.get("Img-Name", row.get("image", f"{idx}.jpg"))
            if pd.isna(img_name):
                continue
            
            img_name = str(img_name).strip()
            image_path = self.images_dir / img_name
            
            # Skip if image doesn't exist
            if not image_path.exists():
                # Try common extensions
                for ext in ['.jpg', '.png', '.jpeg']:
                    alt_path = self.images_dir / f"{Path(img_name).stem}{ext}"
                    if alt_path.exists():
                        image_path = alt_path
                        break
            
            # Get label
            raw_label = row.get(self.label_column)
            label = self._map_label(raw_label)
            
            # Get text if available
            text = row.get("Img-Text")
            if pd.notna(text):
                text = str(text)
            else:
                text = None
            
            samples.append(DatasetSample(
                id=str(idx),
                image_path=str(image_path),
                ground_truth_label=label,
                text=text,
                metadata={"raw_label": raw_label}
            ))
        
        self._samples = samples
        return samples
    
    def get_samples(self, n: Optional[int] = None, shuffle: bool = False) -> List[DatasetSample]:
        """Get samples from the dataset."""
        samples = self.load().copy()
        
        if shuffle:
            import random
            random.shuffle(samples)
        
        if n is not None:
            samples = samples[:n]
        
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        samples = self.load()
        
        hate_count = sum(1 for s in samples if s.ground_truth_label == "HATE")
        non_hate_count = len(samples) - hate_count
        
        return {
            "dataset_name": self.name,
            "total_samples": len(samples),
            "hate_count": hate_count,
            "non_hate_count": non_hate_count,
            "hate_ratio": hate_count / len(samples) if samples else 0
        }
    
    def __len__(self) -> int:
        return len(self.load())
    
    def __iter__(self):
        return iter(self.load())
