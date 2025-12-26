"""Test raw LLaVA output with longer timeout."""
import asyncio
import httpx
import base64
from pathlib import Path

async def test_llava():
    # Get a test image
    image_path = Path("bully_data/0.jpg")
    if not image_path.exists():
        import glob
        images = glob.glob("bully_data/*.jpg")[:1]
        if images:
            image_path = Path(images[0])
        else:
            print("No images found")
            return
    
    print(f"Testing with image: {image_path}")
    
    # Encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Simpler prompt for faster response
    prompt = """Describe this image briefly in JSON format:
{"description": "...", "text_in_image": "...", "is_hateful": true or false}"""

    payload = {
        "model": "llava:7b",
        "prompt": prompt,
        "images": [image_data],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 200  # Shorter response
        }
    }
    
    print("Sending request to Ollama (5 min timeout)...")
    async with httpx.AsyncClient(timeout=300) as client:  # 5 min timeout
        response = await client.post(
            "http://localhost:11434/api/generate",
            json=payload
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            raw_response = result.get("response", "")
            print(f"\n=== RAW LLAVA RESPONSE ===")
            print(raw_response)
            print(f"=== END ===")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_llava())
