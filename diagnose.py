"""Diagnostic script to trace exact error location."""
import sys
sys.path.insert(0, '.')
import os
import asyncio

os.environ['API_PROVIDER'] = 'ollama'
os.environ['VLM_MODEL'] = 'llava:7b'
os.environ['LLM_MODEL'] = 'llama3.2:3b'

from data.datasets.excel_loader import MultiBullyExcelLoader
from src.vlm.ollama_vlm import OllamaVLM
from src.llm.ollama_llm import OllamaLLM
from src.llm.zero_shot import ZeroShotClassifier

async def main():
    # 1. Load one sample
    loader = MultiBullyExcelLoader(
        images_dir='bully_data',
        excel_path='Cyberbully_corrected_emotion_sentiment.xlsx'
    )
    samples = loader.get_samples(n=1, shuffle=True)
    sample = samples[0]
    print(f"Sample: {sample.id}, GT: {sample.ground_truth_label}")
    print(f"Image: {sample.image_path}")
    
    # 2. Test VLM directly
    print("\n=== TESTING VLM ===")
    vlm = OllamaVLM(model="llava:7b")
    try:
        vlm_output = await vlm.analyze_image(str(sample.image_path))
        print(f"VLM SUCCESS!")
        print(f"  Visual: {vlm_output.visual_description[:100]}...")
        print(f"  OCR: {vlm_output.ocr_text[:50] if vlm_output.ocr_text else 'None'}")
        print(f"  Risk: {vlm_output.hate_risk_level}")
    except Exception as e:
        print(f"VLM FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Test LLM directly
    print("\n=== TESTING LLM ===")
    llm = OllamaLLM(model="llama3.2:3b")
    try:
        response = await llm.complete("Say hello in one word.")
        print(f"LLM SUCCESS: {response}")
    except Exception as e:
        print(f"LLM FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Test Classifier
    print("\n=== TESTING CLASSIFIER ===")
    classifier = ZeroShotClassifier(llm)
    try:
        result = await classifier.classify(vlm_output)
        print(f"CLASSIFIER SUCCESS!")
        print(f"  Label: {result.label}")
        print(f"  Justification: {result.justification[:100]}...")
    except Exception as e:
        print(f"CLASSIFIER FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
