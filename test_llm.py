"""Minimal test to check LLM separately."""
import sys
sys.path.insert(0, '.')
import os
import asyncio

os.environ['API_PROVIDER'] = 'ollama'
os.environ['LLM_MODEL'] = 'llama3.2:3b'

from src.llm.ollama_llm import OllamaLLM

async def main():
    print("Testing LLM directly...")
    
    llm = OllamaLLM(model="llama3.2:3b")
    
    print("\n1. Simple test:")
    try:
        response = await llm.complete("Say 'hello' in one word.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("\n2. Classification test:")
    test_prompt = """Classify this content as HATE or NON-HATE:
    
Content: "A meme showing someone making fun of a specific religion."

Respond with JSON only:
{"label": "HATE" or "NON-HATE", "justification": "brief reason"}"""
    
    try:
        response = await llm.complete(test_prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
