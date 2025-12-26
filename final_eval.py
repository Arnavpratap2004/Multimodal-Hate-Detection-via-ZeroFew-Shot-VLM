"""Full end-to-end test with 2 samples."""
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
    print("FULL END-TO-END LOCAL EVALUATION - 2 SAMPLES")
    print("Expect 10-15 minutes per sample")
    print("=" * 60)
    
    loader = MultiBullyExcelLoader(
        images_dir='bully_data',
        excel_path='Cyberbully_corrected_emotion_sentiment.xlsx'
    )
    
    samples = loader.get_samples(n=2, shuffle=True)
    print(f'Loaded {len(samples)} samples')
    
    detector = HateDetector()
    print("[OK] Detector initialized\n")
    
    calculator = MetricsCalculator()
    results = []
    
    for i, sample in enumerate(samples):
        print(f'{"="*60}')
        print(f'SAMPLE {i+1}/{len(samples)}')
        print(f'ID: {sample.id}')
        print(f'GT: {sample.ground_truth_label}')
        print(f'{"="*60}')
        
        start = time.time()
        print(f'[{time.strftime("%H:%M:%S")}] Starting...')
        
        result = await detector.detect(sample.image_path, 'zero_shot')
        
        elapsed = time.time() - start
        print(f'[{time.strftime("%H:%M:%S")}] Done in {elapsed:.0f}s')
        
        if result.error:
            print(f'ERROR: {result.error[:100]}')
        else:
            pred = result.classification.label
            gt = sample.ground_truth_label
            match = "YES" if pred == gt else "NO"
            
            print(f'Prediction: {pred}')
            print(f'Ground Truth: {gt}')
            print(f'Match: {match}')
            print(f'Reason: {result.classification.justification[:100]}...')
            
            calculator.add_result(pred, gt)
            results.append((sample.id, pred, gt, match))
        print()
    
    print(f'{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    for sid, pred, gt, match in results:
        print(f'  {sid}: {pred} vs {gt} -> {match}')
    
    print()
    metrics = calculator.calculate()
    print(f'Accuracy:  {metrics.get("accuracy", 0):.2%}')
    print(f'Precision: {metrics.get("precision", 0):.2%}')
    print(f'Recall:    {metrics.get("recall", 0):.2%}')
    print(f'F1 Score:  {metrics.get("f1_score", 0):.2%}')

if __name__ == '__main__':
    asyncio.run(main())
