"""
Failure mode analysis for hate detection evaluation.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re

from ..pipeline.schemas import FullAnalysis


@dataclass
class FailureCase:
    """A single failure case for analysis."""
    image_path: str
    predicted_label: str
    ground_truth_label: str
    failure_type: str  # "false_positive" or "false_negative"
    vlm_output: Dict[str, Any]
    justification: str
    possible_causes: List[str] = field(default_factory=list)


class FailureModeAnalyzer:
    """
    Analyze failure modes in hate detection to understand model weaknesses.
    
    Focuses on:
    - Code-mixed language failures
    - Sarcasm/humor misclassification
    - Visual-only hate detection failures
    - Cultural reference gaps
    """
    
    # Patterns for code-mixed content
    CODE_MIXED_PATTERNS = [
        r'[\u0900-\u097F]',  # Devanagari (Hindi)
        r'[\u0980-\u09FF]',  # Bengali
        r'[\u0D00-\u0D7F]',  # Malayalam
        r'[\u0B80-\u0BFF]',  # Tamil
    ]
    
    # Keywords suggesting sarcasm
    SARCASM_INDICATORS = [
        "wow", "amazing", "great job", "bravo", "congrats",
        "thank you", "so smart", "genius", "brilliant",
        "waah", "kya baat", "bahut accha", "shabash"
    ]
    
    def __init__(self):
        """Initialize the analyzer."""
        self.failure_cases: List[FailureCase] = []
        self.analysis_complete = False
    
    def add_failure(
        self,
        analysis: FullAnalysis,
        ground_truth: str
    ) -> None:
        """
        Add a failure case for analysis.
        
        Args:
            analysis: The FullAnalysis result from the detector.
            ground_truth: The correct ground truth label.
        """
        if analysis.error:
            return
        
        predicted = analysis.classification.label
        
        if predicted == ground_truth:
            return  # Not a failure
        
        failure_type = "false_positive" if predicted == "HATE" else "false_negative"
        
        case = FailureCase(
            image_path=analysis.image_path,
            predicted_label=predicted,
            ground_truth_label=ground_truth,
            failure_type=failure_type,
            vlm_output=analysis.vlm_output.model_dump(),
            justification=analysis.classification.justification
        )
        
        # Analyze possible causes
        case.possible_causes = self._analyze_causes(analysis, failure_type)
        
        self.failure_cases.append(case)
        self.analysis_complete = False
    
    def _analyze_causes(
        self,
        analysis: FullAnalysis,
        failure_type: str
    ) -> List[str]:
        """
        Analyze possible causes of a failure.
        
        Args:
            analysis: The failed analysis.
            failure_type: Either "false_positive" or "false_negative".
            
        Returns:
            List of possible causes.
        """
        causes = []
        vlm = analysis.vlm_output
        
        # Check for code-mixed content
        combined_text = f"{vlm.ocr_text} {vlm.implicit_meaning}"
        for pattern in self.CODE_MIXED_PATTERNS:
            if re.search(pattern, combined_text):
                causes.append("code_mixed_content")
                break
        
        # Check for sarcasm indicators
        text_lower = combined_text.lower()
        for indicator in self.SARCASM_INDICATORS:
            if indicator in text_lower:
                causes.append("sarcasm_present")
                break
        
        # Check for visual-only potential
        if len(vlm.ocr_text.strip()) < 10:
            causes.append("minimal_text_visual_heavy")
        
        # Check for cultural references
        if "cultural" in vlm.implicit_meaning.lower() or "reference" in vlm.implicit_meaning.lower():
            causes.append("cultural_reference")
        
        # Failure-type specific causes
        if failure_type == "false_negative":
            # Missed hate - might be subtle
            if "subtle" in text_lower or "implicit" in vlm.implicit_meaning.lower():
                causes.append("subtle_hate")
            if vlm.target_group is None:
                causes.append("target_not_identified")
        else:
            # False positive - might be misunderstood humor
            if "humor" in text_lower or "joke" in text_lower:
                causes.append("misunderstood_humor")
            if "satire" in text_lower or "parody" in text_lower:
                causes.append("satire_misclassified")
        
        if not causes:
            causes.append("unknown")
        
        return causes
    
    def get_failure_distribution(self) -> Dict[str, int]:
        """
        Get distribution of failure types.
        
        Returns:
            Dictionary mapping failure types to counts.
        """
        distribution = {
            "false_positives": 0,
            "false_negatives": 0,
            "total": len(self.failure_cases)
        }
        
        for case in self.failure_cases:
            if case.failure_type == "false_positive":
                distribution["false_positives"] += 1
            else:
                distribution["false_negatives"] += 1
        
        return distribution
    
    def get_cause_distribution(self) -> Dict[str, int]:
        """
        Get distribution of failure causes.
        
        Returns:
            Dictionary mapping causes to counts.
        """
        cause_counts = defaultdict(int)
        
        for case in self.failure_cases:
            for cause in case.possible_causes:
                cause_counts[cause] += 1
        
        return dict(sorted(cause_counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_failures_by_cause(self, cause: str) -> List[FailureCase]:
        """
        Get all failure cases with a specific cause.
        
        Args:
            cause: The cause to filter by.
            
        Returns:
            List of matching failure cases.
        """
        return [
            case for case in self.failure_cases
            if cause in case.possible_causes
        ]
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive failure analysis report.
        
        Returns:
            Formatted report string.
        """
        if not self.failure_cases:
            return "No failure cases to analyze."
        
        lines = []
        lines.append("=" * 60)
        lines.append("FAILURE MODE ANALYSIS REPORT")
        lines.append("=" * 60)
        
        # Summary
        dist = self.get_failure_distribution()
        lines.append(f"\nTotal Failures: {dist['total']}")
        lines.append(f"  False Positives: {dist['false_positives']}")
        lines.append(f"  False Negatives: {dist['false_negatives']}")
        
        # Cause distribution
        lines.append("\n" + "-" * 40)
        lines.append("FAILURE CAUSE DISTRIBUTION")
        lines.append("-" * 40)
        
        cause_dist = self.get_cause_distribution()
        for cause, count in cause_dist.items():
            pct = count / len(self.failure_cases) * 100
            cause_label = cause.replace("_", " ").title()
            lines.append(f"  {cause_label}: {count} ({pct:.1f}%)")
        
        # Key insights
        lines.append("\n" + "-" * 40)
        lines.append("KEY INSIGHTS")
        lines.append("-" * 40)
        
        # Code-mixed issues
        code_mixed_count = cause_dist.get("code_mixed_content", 0)
        if code_mixed_count > 0:
            lines.append(f"\n📝 Code-Mixed Content:")
            lines.append(f"   {code_mixed_count} failures involve code-mixed text.")
            lines.append("   Consider using the 'code_mixed' VLM prompt mode for these cases.")
        
        # Sarcasm issues
        sarcasm_count = cause_dist.get("sarcasm_present", 0)
        if sarcasm_count > 0:
            lines.append(f"\n😏 Sarcasm Detection:")
            lines.append(f"   {sarcasm_count} failures involve sarcasm.")
            lines.append("   Sarcastic hate is challenging - consider CoT mode for better reasoning.")
        
        # Visual-heavy issues
        visual_count = cause_dist.get("minimal_text_visual_heavy", 0)
        if visual_count > 0:
            lines.append(f"\n🖼️ Visual-Heavy Content:")
            lines.append(f"   {visual_count} failures have minimal text.")
            lines.append("   The model may be missing visual hate cues.")
        
        # Subtle hate issues
        subtle_count = cause_dist.get("subtle_hate", 0)
        if subtle_count > 0:
            lines.append(f"\n🔍 Subtle Hate:")
            lines.append(f"   {subtle_count} false negatives involve subtle/implicit hate.")
            lines.append("   These cases may benefit from few-shot examples of similar content.")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def get_example_failures(
        self,
        n: int = 5,
        failure_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get example failures for review.
        
        Args:
            n: Number of examples to return.
            failure_type: Filter by "false_positive" or "false_negative".
            
        Returns:
            List of failure case summaries.
        """
        cases = self.failure_cases
        
        if failure_type:
            cases = [c for c in cases if c.failure_type == failure_type]
        
        examples = []
        for case in cases[:n]:
            examples.append({
                "image_path": case.image_path,
                "predicted": case.predicted_label,
                "ground_truth": case.ground_truth_label,
                "failure_type": case.failure_type,
                "ocr_text": case.vlm_output.get("ocr_text", "")[:100],
                "causes": case.possible_causes,
                "justification": case.justification
            })
        
        return examples
    
    def export_for_review(self, filepath: str) -> None:
        """
        Export all failure cases to a JSON file for manual review.
        
        Args:
            filepath: Path to save the JSON file.
        """
        import json
        
        export_data = {
            "summary": self.get_failure_distribution(),
            "cause_distribution": self.get_cause_distribution(),
            "cases": [
                {
                    "image_path": c.image_path,
                    "predicted_label": c.predicted_label,
                    "ground_truth_label": c.ground_truth_label,
                    "failure_type": c.failure_type,
                    "possible_causes": c.possible_causes,
                    "justification": c.justification,
                    "vlm_output": c.vlm_output
                }
                for c in self.failure_cases
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
