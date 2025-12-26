"""Super detailed evaluation to see exact labels."""
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
    print("DETAILED EVALUATION - 1 SAMPLE")
    print("=" * 60)
    
    loader = MultiBullyExcelLoader(
        images_dir='bully_data',
        excel_path='Cyberbully_corrected_emotion_sentiment.xlsx'
    )
    
    samples = loader.get_samples(n=1, shuffle=True)
    sample = samples[0]
    
    print(f'\nSample ID: {sample.id}')
    print(f'Image Path: {sample.image_path}')
    print(f'Ground Truth Label: "{sample.ground_truth_label}"')
    print(f'Ground Truth Type: {type(sample.ground_truth_label)}')
    
    detector = HateDetector()
    print("\n[OK] Detector initialized")
    
    start = time.time()
    print(f'\n[{time.strftime("%H:%M:%S")}] Starting analysis...')
    
    result = await detector.detect(sample.image_path, 'zero_shot')
    
    elapsed = time.time() - start
    print(f'[{time.strftime("%H:%M:%S")}] Completed in {elapsed:.1f}s')
    
    print(f'\n=== RESULT ===')
    print(f'Error: {result.error}')
    print(f'Predicted Label: "{result.classification.label}"')
    print(f'Predicted Label Type: {type(result.classification.label)}')
    print(f'Justification: {result.classification.justification}')
    
    print(f'\n=== COMPARISON ===')
    print(f'Predicted: "{result.classification.label}"')
    print(f'Ground Truth: "{sample.ground_truth_label}"')
    print(f'Match: {result.classification.label == sample.ground_truth_label}')
    
    # Try with MetricsCalculator
    calc = MetricsCalculator()
    calc.add_result(result.classification.label, sample.ground_truth_label)
    metrics = calc.calculate()
    print(f'\n=== METRICS ===')
    print(f'Metrics: {metrics}')

if __name__ == '__main__':
    asyncio.run(main())
