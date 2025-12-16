"""
Zero-shot hate speech classification.

Zero-shot classification uses only the task description without any examples.
This tests the model's inherent understanding of hate speech.
"""

from ..vlm.base import VLMOutput
from .base import BaseClassifier, BaseLLM, ClassificationResult
from .openrouter_llm import OpenRouterLLM


# System prompt for zero-shot classification
ZERO_SHOT_SYSTEM_PROMPT = """You are an expert hate speech analyst specializing in multimodal content analysis.

Your task is to determine whether a meme contains hateful content based on the provided analysis.

## Definition of Hate Speech
Content is considered HATE if it:
1. Contains slurs, derogatory language, or insults targeting specific groups
2. Uses stereotypes to demean, mock, or dehumanize
3. Incites violence, discrimination, or harassment
4. Employs sarcasm, humor, or coded language to convey hate
5. Targets protected groups (religious, ethnic, gender, LGBTQ+, caste, disability, etc.)

## Definition of Non-Hate
Content is NON-HATE if it:
1. Is benign humor without targeting specific groups
2. Makes general observations without malicious intent
3. Criticizes ideas/actions without attacking group identity
4. Contains no derogatory implications even if edgy

## Important Considerations
- Code-mixed language (Hinglish, Bangla-English) may contain subtle slurs
- Sarcasm can be used to convey hate through apparent praise
- Visual elements may reinforce or contradict textual meaning
- Cultural context matters for interpretation

## Response Format
You MUST respond with ONLY a valid JSON object:
{"label": "HATE" or "NON-HATE", "justification": "one sentence explanation"}

Do NOT include any text before or after the JSON.
Do NOT reveal your reasoning process - only the final decision."""


# User prompt template for zero-shot
ZERO_SHOT_PROMPT_TEMPLATE = """Based on the following meme analysis, determine if this content is hateful.

## Meme Analysis

{vlm_context}

## Task
Classify this meme as HATE or NON-HATE and provide a brief justification.

Respond with JSON only: {{"label": "...", "justification": "..."}}"""


class ZeroShotClassifier(BaseClassifier):
    """
    Zero-shot hate speech classifier.
    
    Uses only the task description without examples to classify content.
    This tests the model's inherent understanding of hate speech.
    """
    
    def __init__(self, llm: BaseLLM = None):
        """
        Initialize the zero-shot classifier.
        
        Args:
            llm: The LLM to use for reasoning. Creates default if None.
        """
        if llm is None:
            llm = OpenRouterLLM()
        super().__init__(llm)
    
    @property
    def mode_name(self) -> str:
        """Return the mode name."""
        return "zero_shot"
    
    async def classify(self, vlm_output: VLMOutput) -> ClassificationResult:
        """
        Classify meme content using zero-shot inference.
        
        Args:
            vlm_output: Structured output from VLM image analysis.
            
        Returns:
            ClassificationResult with label and justification.
        """
        # Format VLM output for prompt
        vlm_context = vlm_output.to_context_string()
        
        # Create prompt
        prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(vlm_context=vlm_context)
        
        # Get LLM response
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=ZERO_SHOT_SYSTEM_PROMPT
        )
        
        # Parse response
        try:
            parsed = OpenRouterLLM.parse_json_from_response(response)
            
            label = parsed.get("label", "").upper()
            if label not in ["HATE", "NON-HATE"]:
                # Default to NON-HATE if unclear
                label = "NON-HATE"
            
            justification = parsed.get("justification", "Unable to parse justification")
            
            return ClassificationResult(
                label=label,
                justification=justification
            )
            
        except ValueError as e:
            # If parsing fails, try to extract label from text
            response_upper = response.upper()
            if "HATE" in response_upper and "NON-HATE" not in response_upper:
                label = "HATE"
            else:
                label = "NON-HATE"
            
            return ClassificationResult(
                label=label,
                justification=f"Classification based on response pattern. Original: {response[:100]}..."
            )
