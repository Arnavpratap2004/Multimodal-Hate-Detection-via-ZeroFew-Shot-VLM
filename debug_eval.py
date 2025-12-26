"""Debug evaluation script using local Ollama models."""
import sys
sys.path.insert(0, '.')
import os
import asyncio

# Configure for local Ollama usage
os.environ['API_PROVIDER'] = 'ollama'
os.environ['VLM_MODEL'] = 'llava:7b'
os.environ['LLM_MODEL'] = 'llama3.2:3b'

from data.datasets.excel_loader import MultiBullyExcelLoader
from src.pipeline import HateDetector
from src.evaluation import MetricsCalculator

async def main():
    print("[INFO] Starting Debug Evaluation...")
    
    loader = MultiBullyExcelLoader(
        images_dir='bully_data',
        excel_path='Cyberbully_corrected_emotion_sentiment.xlsx'
    )
    
    samples = loader.get_samples(n=3, shuffle=True)
    print(f'Loaded {len(samples)} samples')
    
    detector = HateDetector()
    print("[INFO] Detector initialized")
    
    calculator = MetricsCalculator()
    errors = 0
    
    for i, sample in enumerate(samples):
        print(f'\n=== SAMPLE {i+1}/{len(samples)} ===')
        print(f'ID: {sample.id}')
        print(f'Image: {sample.image_path}')
        print(f'Ground Truth: {sample.ground_truth_label}')
        
        result = await detector.detect(sample.image_path, 'zero_shot')
        
        print(f'\n--- VLM Output ---')
        print(f'Visual: {result.vlm_output.visual_description[:200]}...')
        print(f'OCR: {result.vlm_output.ocr_text[:100] if result.vlm_output.ocr_text else "None"}')
        print(f'Meaning: {result.vlm_output.implicit_meaning[:100] if result.vlm_output.implicit_meaning else "None"}')
        
        print(f'\n--- Classification ---')
        print(f'Label: {result.classification.label}')
        print(f'Justification: {result.classification.justification[:150]}...')
        
        print(f'\n--- Result ---')
        if result.error:
            print(f'ERROR: {result.error}')
            errors += 1
        else:
            # Show what's being added to calculator
            print(f'Adding to calculator: pred={result.classification.label}, gt={sample.ground_truth_label}')
            calculator.add_result(result.classification.label, sample.ground_truth_label)
            match = 'MATCH' if result.classification.label == sample.ground_truth_label else 'MISMATCH'
            print(f'Result: {match}')
    
    print(f'\n=== FINAL METRICS ===')
    print(f'Total samples: {len(samples)}')
    print(f'Errors: {errors}')
    metrics = calculator.calculate()
    print(f'Metrics dict: {metrics}')
    print(f'Accuracy: {metrics.get("accuracy", 0):.4f}')
    print(f'Precision: {metrics.get("precision", 0):.4f}')
    print(f'Recall: {metrics.get("recall", 0):.4f}')
    print(f'F1: {metrics.get("f1_score", 0):.4f}')

if __name__ == '__main__':
    asyncio.run(main())
