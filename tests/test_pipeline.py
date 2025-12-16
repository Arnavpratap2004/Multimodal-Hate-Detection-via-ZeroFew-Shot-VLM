"""
Tests for the hate detection pipeline.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vlm.base import VLMOutput
from src.llm.base import ClassificationResult
from src.llm.zero_shot import ZeroShotClassifier
from src.llm.few_shot import FewShotClassifier
from src.llm.chain_of_thought import ChainOfThoughtClassifier
from src.pipeline.schemas import FullAnalysis


# Sample VLM output for testing
SAMPLE_VLM_OUTPUT = VLMOutput(
    visual_description="A meme showing two people in conversation with text overlays",
    ocr_text="When they say 'all of them are the same'",
    implicit_meaning="Generalizing stereotype about a group, implying all members share negative traits",
    target_group="Unspecified religious group"
)

BENIGN_VLM_OUTPUT = VLMOutput(
    visual_description="A cute cat looking confused at a salad",
    ocr_text="Me trying to eat healthy",
    implicit_meaning="Relatable humor about difficulty of dieting",
    target_group=None
)


class TestVLMOutput:
    """Tests for VLMOutput model."""
    
    def test_vlm_output_creation(self):
        """Test creating a VLMOutput instance."""
        output = VLMOutput(
            visual_description="Test description",
            ocr_text="Test text",
            implicit_meaning="Test meaning",
            target_group="Test group"
        )
        
        assert output.visual_description == "Test description"
        assert output.ocr_text == "Test text"
        assert output.implicit_meaning == "Test meaning"
        assert output.target_group == "Test group"
    
    def test_vlm_output_optional_target(self):
        """Test VLMOutput with None target group."""
        output = VLMOutput(
            visual_description="Test",
            ocr_text="Test",
            implicit_meaning="Test",
            target_group=None
        )
        
        assert output.target_group is None
    
    def test_to_context_string(self):
        """Test converting VLMOutput to context string."""
        context = SAMPLE_VLM_OUTPUT.to_context_string()
        
        assert "Visual Description" in context
        assert "OCR" in context
        assert "Implicit Meaning" in context
        assert "Target Group" in context


class TestClassificationResult:
    """Tests for ClassificationResult model."""
    
    def test_hate_classification(self):
        """Test creating a HATE classification."""
        result = ClassificationResult(
            label="HATE",
            justification="Contains derogatory stereotypes"
        )
        
        assert result.label == "HATE"
        assert "stereotype" in result.justification.lower()
    
    def test_non_hate_classification(self):
        """Test creating a NON-HATE classification."""
        result = ClassificationResult(
            label="NON-HATE",
            justification="Benign humor without targeting groups"
        )
        
        assert result.label == "NON-HATE"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ClassificationResult(
            label="HATE",
            justification="Test",
            confidence=0.95
        )
        
        d = result.to_dict()
        assert d["label"] == "HATE"
        assert d["justification"] == "Test"
        assert d["confidence"] == 0.95


class TestZeroShotClassifier:
    """Tests for ZeroShotClassifier."""
    
    @pytest.mark.asyncio
    async def test_classify_with_mock_llm(self):
        """Test classification with mocked LLM."""
        # Create mock LLM
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = '{"label": "HATE", "justification": "Contains stereotypes"}'
        
        classifier = ZeroShotClassifier(llm=mock_llm)
        result = await classifier.classify(SAMPLE_VLM_OUTPUT)
        
        assert result.label == "HATE"
        assert mock_llm.complete.called
    
    @pytest.mark.asyncio
    async def test_classify_benign_content(self):
        """Test classifying benign content."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = '{"label": "NON-HATE", "justification": "Harmless humor"}'
        
        classifier = ZeroShotClassifier(llm=mock_llm)
        result = await classifier.classify(BENIGN_VLM_OUTPUT)
        
        assert result.label == "NON-HATE"
    
    def test_mode_name(self):
        """Test mode name property."""
        mock_llm = MagicMock()
        classifier = ZeroShotClassifier(llm=mock_llm)
        assert classifier.mode_name == "zero_shot"


class TestFewShotClassifier:
    """Tests for FewShotClassifier."""
    
    @pytest.mark.asyncio
    async def test_classify_with_examples(self):
        """Test classification using few-shot examples."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = '{"label": "HATE", "justification": "Similar to example 2"}'
        
        classifier = FewShotClassifier(llm=mock_llm, num_examples=3)
        result = await classifier.classify(SAMPLE_VLM_OUTPUT)
        
        assert result.label == "HATE"
        
        # Check that examples were included in prompt
        call_args = mock_llm.complete.call_args
        prompt = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
        assert "Example" in prompt
    
    def test_mode_name(self):
        """Test mode name property."""
        mock_llm = MagicMock()
        classifier = FewShotClassifier(llm=mock_llm)
        assert classifier.mode_name == "few_shot"


class TestChainOfThoughtClassifier:
    """Tests for ChainOfThoughtClassifier."""
    
    @pytest.mark.asyncio
    async def test_classify_with_reasoning(self):
        """Test CoT classification with reasoning."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = '''
        <thinking>
        Step 1: Target is religious group
        Step 2: Generalizing statement
        Step 3: No explicit sarcasm but implied negative
        Step 4: Visual neutral
        Step 5: Cultural context suggests hate
        Step 6: HATE - stereotyping
        </thinking>
        
        <output>
        {"label": "HATE", "justification": "Generalizing stereotype about religious group"}
        </output>
        '''
        
        classifier = ChainOfThoughtClassifier(llm=mock_llm)
        result = await classifier.classify(SAMPLE_VLM_OUTPUT)
        
        assert result.label == "HATE"
    
    def test_mode_name(self):
        """Test mode name property."""
        mock_llm = MagicMock()
        classifier = ChainOfThoughtClassifier(llm=mock_llm)
        assert classifier.mode_name == "cot"


class TestFullAnalysis:
    """Tests for FullAnalysis schema."""
    
    def test_full_analysis_creation(self):
        """Test creating a FullAnalysis instance."""
        classification = ClassificationResult(
            label="HATE",
            justification="Test"
        )
        
        analysis = FullAnalysis(
            image_path="/path/to/image.jpg",
            vlm_output=SAMPLE_VLM_OUTPUT,
            classification=classification,
            inference_mode="zero_shot",
            processing_time=1.5
        )
        
        assert analysis.image_path == "/path/to/image.jpg"
        assert analysis.inference_mode == "zero_shot"
        assert analysis.processing_time == 1.5
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        classification = ClassificationResult(
            label="NON-HATE",
            justification="Benign"
        )
        
        analysis = FullAnalysis(
            image_path="/test.jpg",
            vlm_output=BENIGN_VLM_OUTPUT,
            classification=classification,
            inference_mode="cot",
            processing_time=2.0
        )
        
        d = analysis.to_dict()
        assert d["image_path"] == "/test.jpg"
        assert d["classification"]["label"] == "NON-HATE"
    
    def test_summary(self):
        """Test generating summary string."""
        classification = ClassificationResult(
            label="HATE",
            justification="Stereotypes present"
        )
        
        analysis = FullAnalysis(
            image_path="/test.jpg",
            vlm_output=SAMPLE_VLM_OUTPUT,
            classification=classification,
            inference_mode="few_shot",
            processing_time=1.0
        )
        
        summary = analysis.summary()
        assert "HATE" in summary
        assert "few_shot" in summary


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
