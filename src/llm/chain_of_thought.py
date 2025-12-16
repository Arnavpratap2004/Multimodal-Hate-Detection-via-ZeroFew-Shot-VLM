"""
Chain-of-Thought (CoT) hate speech classification.

CoT classification guides the model through explicit reasoning steps
before making a final classification. The reasoning is internal and
not exposed in the final output.
"""

from ..vlm.base import VLMOutput
from .base import BaseClassifier, BaseLLM, ClassificationResult
from .openrouter_llm import OpenRouterLLM


# System prompt for CoT classification
COT_SYSTEM_PROMPT = """You are an expert hate speech analyst. Your task is to classify memes using careful step-by-step reasoning.

## Analysis Framework
For each meme, you MUST work through these reasoning steps:

### Step 1: Target Identification
- Who or what group is being referenced?
- Is this a protected group (religious, ethnic, gender, LGBTQ+, caste, disability)?
- Is the target explicit or implied?

### Step 2: Language Analysis
- Are there slurs, derogatory terms, or coded language?
- Is the language abusive, mocking, or demeaning?
- For code-mixed text, are there hidden slurs in the non-English portions?

### Step 3: Sarcasm & Irony Detection
- Is sarcasm being used to convey hate through apparent praise?
- Is irony masking harmful stereotypes?
- Does the tone contradict the literal meaning?

### Step 4: Visual-Text Alignment
- Do visual elements reinforce or contradict the text?
- Are visual stereotypes or caricatures present?
- Is there violent or threatening imagery?

### Step 5: Cultural Context
- Are there culture-specific references that convey hate?
- Historical context that makes the content harmful?
- Regional stereotypes being invoked?

### Step 6: Final Determination
Based on all above steps:
- HATE: Content attacks, demeans, or threatens based on group identity
- NON-HATE: Content is benign, general criticism, or doesn't target groups

## Response Format
You MUST respond in this EXACT format:

<thinking>
[Your step-by-step analysis - be thorough]
</thinking>

<output>
{"label": "HATE" or "NON-HATE", "justification": "one sentence summary"}
</output>

IMPORTANT: Only the content inside <output> tags will be used for the final result."""


# Prompt template for CoT
COT_PROMPT_TEMPLATE = """Analyze the following meme using careful step-by-step reasoning.

## Meme Analysis

**Visual Description:**
{visual_description}

**OCR Text:**
{ocr_text}

**Implicit Meaning:**
{implicit_meaning}

**Identified Target Group:**
{target_group}

## Instructions
1. Work through all 6 reasoning steps in the <thinking> section
2. Be thorough in your analysis
3. Provide your final answer in the <output> section as JSON

Begin your analysis:"""


class ChainOfThoughtClassifier(BaseClassifier):
    """
    Chain-of-Thought hate speech classifier.
    
    Guides the model through explicit reasoning steps for more
    robust classification, especially for subtle or complex cases.
    """
    
    def __init__(self, llm: BaseLLM = None):
        """
        Initialize the CoT classifier.
        
        Args:
            llm: The LLM to use for reasoning. Creates default if None.
        """
        if llm is None:
            llm = OpenRouterLLM(temperature=0.2)  # Slightly higher for reasoning
        super().__init__(llm)
    
    @property
    def mode_name(self) -> str:
        """Return the mode name."""
        return "cot"
    
    def _extract_output_section(self, response: str) -> str:
        """
        Extract the content between <output> tags.
        
        Args:
            response: Full LLM response with thinking and output sections.
            
        Returns:
            The content inside <output> tags.
        """
        # Try to find <output> tags
        import re
        
        # Pattern to match <output>...</output>
        pattern = r'<output>\s*(.*?)\s*</output>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: try to find JSON at the end
        # Remove thinking section if present
        if '<thinking>' in response.lower():
            parts = re.split(r'</thinking>', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[-1]
        
        return response
    
    async def classify(self, vlm_output: VLMOutput) -> ClassificationResult:
        """
        Classify meme content using chain-of-thought reasoning.
        
        Args:
            vlm_output: Structured output from VLM image analysis.
            
        Returns:
            ClassificationResult with label and justification.
        """
        # Format prompt
        prompt = COT_PROMPT_TEMPLATE.format(
            visual_description=vlm_output.visual_description,
            ocr_text=vlm_output.ocr_text,
            implicit_meaning=vlm_output.implicit_meaning,
            target_group=vlm_output.target_group or "None identified"
        )
        
        # Get LLM response (includes thinking + output)
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=COT_SYSTEM_PROMPT
        )
        
        # Extract just the output section
        output_content = self._extract_output_section(response)
        
        # Parse the JSON output
        try:
            parsed = OpenRouterLLM.parse_json_from_response(output_content)
            
            label = parsed.get("label", "").upper()
            if label not in ["HATE", "NON-HATE"]:
                label = "NON-HATE"
            
            justification = parsed.get("justification", "Unable to parse justification")
            
            return ClassificationResult(
                label=label,
                justification=justification
            )
            
        except ValueError:
            # Fallback: try to extract from full response
            response_upper = response.upper()
            
            # Look for explicit label mentions
            if '"HATE"' in response or "'HATE'" in response:
                if '"NON-HATE"' not in response and "'NON-HATE'" not in response:
                    label = "HATE"
                else:
                    # Both present, look at context
                    hate_pos = response_upper.rfind('"HATE"')
                    non_hate_pos = response_upper.rfind('"NON-HATE"')
                    label = "NON-HATE" if non_hate_pos > hate_pos else "HATE"
            else:
                label = "NON-HATE"
            
            return ClassificationResult(
                label=label,
                justification="Classification based on chain-of-thought reasoning."
            )


class EnhancedCoTClassifier(ChainOfThoughtClassifier):
    """
    Enhanced CoT classifier with additional reasoning for edge cases.
    
    Adds specialized prompts for code-mixed content and visual-only hate.
    """
    
    async def classify_with_focus(
        self,
        vlm_output: VLMOutput,
        focus: str = "general"
    ) -> ClassificationResult:
        """
        Classify with a specific analytical focus.
        
        Args:
            vlm_output: Structured output from VLM image analysis.
            focus: One of "general", "code_mixed", "visual", "sarcasm"
            
        Returns:
            ClassificationResult with label and justification.
        """
        focus_additions = {
            "code_mixed": "\n\n**SPECIAL FOCUS**: Pay extra attention to code-mixed language. Check for slurs, derogatory terms, or hate speech hidden in non-English portions of the text.",
            "visual": "\n\n**SPECIAL FOCUS**: Pay extra attention to visual elements. Even if text seems benign, check for hateful imagery, stereotypical depictions, or visual dog-whistles.",
            "sarcasm": "\n\n**SPECIAL FOCUS**: Pay extra attention to sarcasm and irony. The literal meaning may seem positive but could convey hate through mockery or backhanded compliments."
        }
        
        # Add focus instruction to VLM output
        if focus in focus_additions:
            original_meaning = vlm_output.implicit_meaning
            vlm_output = VLMOutput(
                visual_description=vlm_output.visual_description,
                ocr_text=vlm_output.ocr_text,
                implicit_meaning=original_meaning + focus_additions[focus],
                target_group=vlm_output.target_group
            )
        
        return await self.classify(vlm_output)
