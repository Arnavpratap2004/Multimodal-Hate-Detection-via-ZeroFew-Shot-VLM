"""Test different prompt formats with LLaVA to find one that produces parsable JSON."""
import asyncio
import httpx
import base64
import json
from pathlib import Path

# Test prompts - from most constrained to least
PROMPTS = {
    "minimal": """Look at this image. Answer these 4 questions:
1. What do you see?
2. What text is in the image?
3. What does it mean?
4. Is it offensive? (yes/no)

Reply in this exact format:
{"see": "answer1", "text": "answer2", "meaning": "answer3", "offensive": "yes or no"}""",

    "structured": """Analyze this meme. Reply ONLY with JSON, no other text:

{
  "description": "describe what you see",
  "text_in_image": "copy any text you see", 
  "is_hateful": true or false
}""",

    "step_by_step": """Step 1: Describe what you see in this image.
Step 2: Copy any text visible in the image.
Step 3: Explain what the meme means.
Step 4: Could this offend someone? Answer HIGH, MEDIUM, or LOW.

Format your answer as JSON:
{"visual": "step1", "ocr": "step2", "meaning": "step3", "risk": "step4"}""",

    "explicit_json": """OUTPUT FORMAT: You must respond with a valid JSON object only. No markdown, no explanation.

Analyze this meme image:

{
  "visual_description": "[describe the image]",
  "ocr_text": "[text from image or empty string]",
  "implicit_meaning": "[what the meme communicates]",
  "hate_risk_level": "LOW or MEDIUM or HIGH"
}

Replace the bracketed placeholders with your analysis. Output ONLY the JSON."""
}

async def test_prompt(prompt_name: str, prompt: str, image_data: str):
    """Test a single prompt."""
    payload = {
        "model": "llava:7b",
        "prompt": prompt,
        "images": [image_data],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300
        }
    }
    
    print(f"\n{'='*50}")
    print(f"Testing: {prompt_name}")
    print(f"{'='*50}")
    
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json=payload
        )
        
        if response.status_code != 200:
            print(f"ERROR: {response.text}")
            return None
        
        result = response.json()
        raw = result.get("response", "")
        print(f"Raw response:\n{raw}\n")
        
        # Try to parse as JSON
        try:
            # Clean up response
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            parsed = json.loads(cleaned)
            print(f"SUCCESS: Parsed as JSON!")
            print(f"Keys: {list(parsed.keys())}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"FAILED to parse: {e}")
            return None

async def main():
    # Get test image
    image_path = Path("bully_data/0.jpg")
    if not image_path.exists():
        import glob
        images = glob.glob("bully_data/*.jpg")[:1]
        if images:
            image_path = Path(images[0])
    
    print(f"Using image: {image_path}")
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    results = {}
    for name, prompt in PROMPTS.items():
        try:
            result = await test_prompt(name, prompt, image_data)
            results[name] = "SUCCESS" if result else "FAILED"
        except Exception as e:
            print(f"ERROR with {name}: {e}")
            results[name] = "ERROR"
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, status in results.items():
        print(f"  {name}: {status}")

if __name__ == "__main__":
    asyncio.run(main())
