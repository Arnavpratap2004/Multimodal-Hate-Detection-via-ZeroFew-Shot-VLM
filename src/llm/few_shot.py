"""
Few-shot hate speech classification.

Few-shot classification provides curated examples to help the model
understand the classification task better.
"""

from typing import List, Optional
from pydantic import BaseModel

from ..vlm.base import VLMOutput
from .base import BaseClassifier, BaseLLM, ClassificationResult
from .openrouter_llm import OpenRouterLLM


class FewShotExample(BaseModel):
    """A labeled example for few-shot learning."""
    
    visual_description: str
    ocr_text: str
    implicit_meaning: str
    target_group: Optional[str]
    label: str  # "HATE" or "NON-HATE"
    justification: str


# Default curated examples for few-shot learning
DEFAULT_EXAMPLES: List[FewShotExample] = [
    # Example 1: Clear hate speech (anti-religious)
    FewShotExample(
        visual_description="An image showing a person of a specific religious group depicted as a caricature with exaggerated features, holding a weapon",
        ocr_text="All of them are terrorists",
        implicit_meaning="The meme stereotypes an entire religious community as violent terrorists, using visual caricature to dehumanize",
        target_group="Religious minority group",
        label="HATE",
        justification="Promotes harmful stereotype that an entire religious group is violent and dangerous"
    ),
    
    # Example 2: Subtle sarcastic hate
    FewShotExample(
        visual_description="Two panel meme showing contrasting images - one of a successful person, one showing crime scene photos",
        ocr_text="Before reservation vs After reservation. Such progress!",
        implicit_meaning="Sarcastic comment implying that affirmative action policies have led to negative outcomes, targeting specific caste groups",
        target_group="Lower caste communities benefiting from reservation",
        label="HATE",
        justification="Uses sarcasm to mock affirmative action and implicitly demean caste-based minority groups"
    ),
    
    # Example 3: Code-mixed hate (Hinglish)
    FewShotExample(
        visual_description="Image of a person with derogatory makeup making exaggerated expressions",
        ocr_text="Ye log kaise bhi dikhe, inside sab same hi hote hain 🤡",
        implicit_meaning="Code-mixed text suggesting all members of a group are the same (negative), using clown emoji to mock",
        target_group="Specific ethnic/regional group",
        label="HATE",
        justification="Stereotypes an entire group as having negative traits regardless of appearance"
    ),
    
    # Example 4: Non-hate (benign humor)
    FewShotExample(
        visual_description="Cat sitting at a table looking confused, classic meme format",
        ocr_text="When you realize Monday is tomorrow",
        implicit_meaning="Relatable humor about disliking Mondays, universal experience",
        target_group=None,
        label="NON-HATE",
        justification="Generic relatable humor without targeting any group or conveying hate"
    ),
    
    # Example 5: Non-hate (political criticism without hate)
    FewShotExample(
        visual_description="Politician speaking at podium with speech bubble added",
        ocr_text="Trust me, this policy will definitely work this time",
        implicit_meaning="Sarcastic criticism of a political figure's track record on policy",
        target_group=None,
        label="NON-HATE",
        justification="Political satire criticizing policy/actions, not attacking identity or a protected group"
    ),
]


# System prompt for few-shot classification
FEW_SHOT_SYSTEM_PROMPT = """You are an expert hate speech analyst. You will be shown examples of meme analyses with their correct classifications, then asked to classify a new meme.

## Task
Learn from the provided examples to classify new memes as HATE or NON-HATE.

## Definition of HATE
- Contains slurs, derogatory language targeting groups
- Uses stereotypes to demean or dehumanize
- Incites violence or discrimination
- Employs sarcasm to convey hate

## Definition of NON-HATE
- Benign humor without targeting groups
- Political criticism of actions, not identity
- General observations without malice

## Response Format
Respond with ONLY a valid JSON object:
{"label": "HATE" or "NON-HATE", "justification": "one sentence explanation"}"""


def format_example(example: FewShotExample, index: int) -> str:
    """Format a single example for the prompt."""
    target = example.target_group or "None"
    return f"""### Example {index}
**Visual Description:** {example.visual_description}
**OCR Text:** {example.ocr_text}
**Implicit Meaning:** {example.implicit_meaning}
**Target Group:** {target}

**Classification:** {example.label}
**Justification:** {example.justification}
"""


class FewShotClassifier(BaseClassifier):
    """
    Few-shot hate speech classifier.
    
    Uses curated examples to guide the model's classification.
    """
    
    def __init__(
        self,
        llm: BaseLLM = None,
        examples: Optional[List[FewShotExample]] = None,
        num_examples: int = 5
    ):
        """
        Initialize the few-shot classifier.
        
        Args:
            llm: The LLM to use for reasoning. Creates default if None.
            examples: Custom examples to use. Uses defaults if None.
            num_examples: Number of examples to include in prompt.
        """
        if llm is None:
            llm = OpenRouterLLM()
        super().__init__(llm)
        
        self.examples = examples or DEFAULT_EXAMPLES
        self.num_examples = min(num_examples, len(self.examples))
    
    @property
    def mode_name(self) -> str:
        """Return the mode name."""
        return "few_shot"
    
    def _build_examples_section(self) -> str:
        """Build the examples section of the prompt."""
        examples_text = "## Training Examples\n\n"
        for i, example in enumerate(self.examples[:self.num_examples], 1):
            examples_text += format_example(example, i)
            examples_text += "---\n\n"
        return examples_text
    
    async def classify(self, vlm_output: VLMOutput) -> ClassificationResult:
        """
        Classify meme content using few-shot inference.
        
        Args:
            vlm_output: Structured output from VLM image analysis.
            
        Returns:
            ClassificationResult with label and justification.
        """
        # Build prompt with examples
        examples_section = self._build_examples_section()
        
        target = vlm_output.target_group or "None"
        
        prompt = f"""{examples_section}

## New Meme to Classify

**Visual Description:** {vlm_output.visual_description}
**OCR Text:** {vlm_output.ocr_text}
**Implicit Meaning:** {vlm_output.implicit_meaning}
**Target Group:** {target}

Based on the examples above, classify this meme.

Respond with JSON only: {{"label": "...", "justification": "..."}}"""
        
        # Get LLM response
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=FEW_SHOT_SYSTEM_PROMPT
        )
        
        # Parse response
        try:
            parsed = OpenRouterLLM.parse_json_from_response(response)
            
            label = parsed.get("label", "").upper()
            if label not in ["HATE", "NON-HATE"]:
                label = "NON-HATE"
            
            justification = parsed.get("justification", "Unable to parse justification")
            
            return ClassificationResult(
                label=label,
                justification=justification
            )
            
        except ValueError:
            response_upper = response.upper()
            if "HATE" in response_upper and "NON-HATE" not in response_upper:
                label = "HATE"
            else:
                label = "NON-HATE"
            
            return ClassificationResult(
                label=label,
                justification=f"Classification based on response pattern."
            )
    
    def add_example(self, example: FewShotExample) -> None:
        """Add a new example to the classifier."""
        self.examples.append(example)
    
    def set_examples(self, examples: List[FewShotExample]) -> None:
        """Replace all examples."""
        self.examples = examples
