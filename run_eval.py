"""Quick evaluation script using local Ollama models."""
import sys
sys.path.insert(0, '.')
import os
import asyncio

# Configure for local Ollama usage
os.environ['API_PROVIDER'] = 'ollama'
os.environ['VLM_MODEL'] = 'llava:7b'
os.environ['LLM_MODEL'] = 'llama3.2:3b'

# Set encoding to utf-8 for console output
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

from data.datasets.excel_loader import MultiBullyExcelLoader
from src.pipeline import HateDetector
from src.evaluation import MetricsCalculator

async def main():
    print("[INFO] Starting Local Evaluation with Ollama...")
    
    loader = MultiBullyExcelLoader(
        images_dir='bully_data',
        excel_path='Cyberbully_corrected_emotion_sentiment.xlsx'
    )
    
    # Run a small batch first to verify
    samples = loader.get_samples(n=5, shuffle=True)
    print(f'Loaded {len(samples)} samples to test')
    
    # Initialize detector (will auto-use Ollama based on env var)
    detector = HateDetector()
    print("[INFO] Detector initialized with local models")
    
    calculator = MetricsCalculator()
    errors = 0
    
    for i, sample in enumerate(samples):
        print(f'\nProcessing {i+1}/{len(samples)} - ID: {sample.id}')
        result = await detector.detect(sample.image_path, 'zero_shot')
        
        if result.error:
            print(f'  [ERROR]: {result.error[:100]}')
            errors += 1
        else:
            calculator.add_result(result.classification.label, sample.ground_truth_label)
            match = 'MATCH' if result.classification.label == sample.ground_truth_label else 'MISMATCH'
            print(f'  GT: {sample.ground_truth_label} | Pred: {result.classification.label} | {match}')
            print(f'  Visual: {result.vlm_output.visual_description[:100]}...')
            print(f'  Reasoning: {result.classification.justification[:100]}...')
    
    print(f'\n=== RESULTS ===')
    print(f'Errors: {errors}/{len(samples)}')
    metrics = calculator.calculate()
    print(f'Accuracy: {metrics.get("accuracy", 0):.4f}')
    print(f'Precision: {metrics.get("precision", 0):.4f}')
    print(f'Recall: {metrics.get("recall", 0):.4f}')
    print(f'F1: {metrics.get("f1_score", 0):.4f}')

if __name__ == '__main__':
    asyncio.run(main())
