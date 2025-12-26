"""Patient evaluation with 1 sample and extended timeout."""
import sys
sys.path.insert(0, '.')
import os
import asyncio
import time

os.environ['API_PROVIDER'] = 'ollama'
os.environ['VLM_MODEL'] = 'llava:7b'
os.environ['LLM_MODEL'] = 'llama3.2:3b'

from data.datasets.excel_loader import MultiBullyExcelLoader
from src.pipeline import HateDetector
from src.evaluation import MetricsCalculator

async def main():
    print("=" * 60)
    print("PATIENT LOCAL EVALUATION")
    print("Using LLaVA 7B on CPU - expect 5-10 minutes per image")
    print("=" * 60)
    
    loader = MultiBullyExcelLoader(
        images_dir='bully_data',
        excel_path='Cyberbully_corrected_emotion_sentiment.xlsx'
    )
    
    # Only 1 sample for initial test
    samples = loader.get_samples(n=1, shuffle=True)
    print(f'\nLoaded {len(samples)} sample(s)')
    
    detector = HateDetector()
    print("[OK] Detector initialized with local Ollama models")
    
    calculator = MetricsCalculator()
    errors = 0
    
    for i, sample in enumerate(samples):
        print(f'\n{"="*60}')
        print(f'SAMPLE {i+1}/{len(samples)}')
        print(f'ID: {sample.id}')
        print(f'Image: {sample.image_path}')
        print(f'Ground Truth: {sample.ground_truth_label}')
        print(f'{"="*60}')
        
        start = time.time()
        print(f'[{time.strftime("%H:%M:%S")}] Starting analysis (please wait 5-10 minutes)...')
        
        result = await detector.detect(sample.image_path, 'zero_shot')
        
        elapsed = time.time() - start
        print(f'[{time.strftime("%H:%M:%S")}] Completed in {elapsed:.1f} seconds')
        
        if result.error:
            print(f'\nERROR: {result.error}')
            errors += 1
        else:
            print(f'\n--- VLM OUTPUT ---')
            print(f'Visual: {result.vlm_output.visual_description[:200]}...')
            print(f'OCR: {result.vlm_output.ocr_text[:100] if result.vlm_output.ocr_text else "None"}')
            print(f'Risk Level: {result.vlm_output.hate_risk_level}')
            
            print(f'\n--- CLASSIFICATION ---')
            print(f'Predicted: {result.classification.label}')
            print(f'Ground Truth: {sample.ground_truth_label}')
            print(f'Match: {"YES" if result.classification.label == sample.ground_truth_label else "NO"}')
            print(f'Justification: {result.classification.justification[:150]}...')
            
            calculator.add_result(result.classification.label, sample.ground_truth_label)
    
    print(f'\n{"="*60}')
    print('FINAL RESULTS')
    print(f'{"="*60}')
    print(f'Samples processed: {len(samples)}')
    print(f'Errors: {errors}')
    
    if len(samples) - errors > 0:
        metrics = calculator.calculate()
        print(f'Accuracy: {metrics.get("accuracy", 0):.4f}')
        print(f'Precision: {metrics.get("precision", 0):.4f}')
        print(f'Recall: {metrics.get("recall", 0):.4f}')
        print(f'F1: {metrics.get("f1_score", 0):.4f}')

if __name__ == '__main__':
    asyncio.run(main())
