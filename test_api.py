"""Test OpenRouter API connectivity."""
import os
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_api():
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"API Key (first 15 chars): {api_key[:15] if api_key else 'NOT SET'}...")
    
    # Test 1: List models (this should work without credits)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"\nModels API: Status {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()
            # Find free models with vision support
            free_vision_models = []
            for m in models.get('data', []):
                model_id = m.get('id', '')
                pricing = m.get('pricing', {})
                prompt_price = float(pricing.get('prompt', '1') or '1')
                
                # Check if free (price = 0) and has image support
                modalities = m.get('architecture', {}).get('modality', '')
                if prompt_price == 0 and 'image' in modalities.lower():
                    free_vision_models.append(model_id)
            
            print(f"\nFree Vision Models Found: {len(free_vision_models)}")
            for m in free_vision_models[:10]:
                print(f"  - {m}")
        else:
            print(f"Error: {response.text}")

if __name__ == '__main__':
    asyncio.run(test_api())
