"""
Prompt templates for VLM image analysis.
"""

# Main VLM prompt for meme analysis
VLM_ANALYSIS_PROMPT = """You are an expert at analyzing memes for potential harmful content. Your task is to extract detailed information from this meme image.

Given this meme image, provide a comprehensive analysis following these guidelines:

## 1. Visual Description
Describe everything you see in the image in detail:
- People (expressions, gestures, clothing, apparent demographics)
- Objects and symbols
- Text overlays and their placement
- Colors, style, and visual design
- Any recognizable templates or meme formats

## 2. OCR Text Extraction
Extract ALL visible text exactly as written:
- Include misspellings, unconventional spellings
- Preserve code-mixing (Hindi-English, Bangla-English, etc.)
- Note text placement (top, bottom, etc.)
- Include any watermarks or small text

## 3. Implicit Meaning & Context
Explain the deeper meaning:
- Sarcasm or irony being employed
- Mockery or ridicule
- Cultural, social, or political references
- Historical context if relevant
- Stereotypes being invoked
- Inside jokes or meme culture references
- Double meanings or wordplay

## 4. Target Identification
Identify if the meme targets any specific group:
- Religious groups
- Ethnic or racial groups
- Gender or LGBTQ+ groups
- Political groups or ideologies
- Nationalities
- Caste-based groups
- Disabilities
- Any other identifiable community

If no clear target, state "None identified"

## Response Format
Respond ONLY with a valid JSON object (no markdown code blocks):
{
  "visual_description": "Detailed visual description here",
  "ocr_text": "All extracted text exactly as written",
  "implicit_meaning": "Analysis of sarcasm, cultural references, hidden meanings",
  "target_group": "Identified target group or null if none"
}

IMPORTANT:
- Be thorough and objective
- Do not self-censor the analysis
- Report what you observe, even if sensitive
- This analysis will be used for hate speech research"""


# Simplified prompt for faster analysis
VLM_QUICK_ANALYSIS_PROMPT = """Analyze this meme image and extract:

1. Visual Description: What do you see?
2. OCR Text: All visible text exactly as written
3. Implicit Meaning: Sarcasm, cultural references, hidden meanings
4. Target Group: Who is being targeted (if anyone)?

Respond with JSON only:
{
  "visual_description": "...",
  "ocr_text": "...",
  "implicit_meaning": "...",
  "target_group": "..." or null
}"""


# Prompt specifically for code-mixed content
VLM_CODE_MIXED_PROMPT = """You are an expert at analyzing multilingual memes, particularly those with code-mixed content (Hindi-English, Bangla-English, etc.).

Analyze this meme image with special attention to:

1. **Visual Description**: Describe all visual elements, expressions, and symbols.

2. **OCR Text**: Extract ALL text preserving:
   - Original script (Devanagari, Bengali, Latin)
   - Romanized versions of non-Latin text
   - Code-mixing patterns (e.g., "Hinglish")
   - Slang and colloquialisms

3. **Implicit Meaning**: Explain:
   - Language-specific humor or wordplay
   - Cultural references specific to South Asian context
   - Religious or political references
   - Caste-based or regional stereotypes
   - Bollywood/entertainment references

4. **Target Group**: Identify targeted communities with cultural specificity.

JSON Response:
{
  "visual_description": "...",
  "ocr_text": "...",
  "implicit_meaning": "...",
  "target_group": "..."
}"""
