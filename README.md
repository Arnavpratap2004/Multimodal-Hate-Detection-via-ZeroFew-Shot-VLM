# Multimodal Hate Detection via Zero/Few-Shot VLMs

A research-grade multimodal hate detection system using Vision-Language Models (VLMs) and Large Language Models (LLMs) for zero-shot, few-shot, and chain-of-thought inference on meme datasets.

## 🎯 Overview

This project implements a hate detection pipeline following the strict constraint:

```
Image → VLM → Image Description + OCR → LLM → HATE/NON-HATE
```

**Key Features:**
- 🔍 Zero-shot, few-shot, and chain-of-thought classification
- 🌐 Multilingual support (Hindi-English, Bangla-English code-mixing)
- 📊 Comprehensive evaluation framework
- 🚫 **No training or fine-tuning** - inference only

## 📁 Project Structure

```
multimodal_hate_detection/
├── src/
│   ├── config.py              # Configuration and API keys
│   ├── vlm/                   # Vision-Language Model module
│   │   ├── base.py            # Abstract VLM interface
│   │   ├── openrouter_vlm.py  # OpenRouter implementation
│   │   └── prompts.py         # VLM prompts
│   ├── llm/                   # LLM Reasoning module
│   │   ├── base.py            # Abstract LLM interface
│   │   ├── openrouter_llm.py  # OpenRouter implementation
│   │   ├── zero_shot.py       # Zero-shot classifier
│   │   ├── few_shot.py        # Few-shot classifier
│   │   └── chain_of_thought.py # CoT classifier
│   ├── pipeline/              # Main detection pipeline
│   │   ├── detector.py        # Orchestrator
│   │   └── schemas.py         # Pydantic models
│   └── evaluation/            # Metrics and analysis
│       ├── metrics.py
│       └── analyzer.py
├── data/
│   ├── samples/               # Test memes
│   └── datasets/              # Dataset loaders
├── notebooks/
│   └── evaluation.ipynb
├── scripts/
│   └── evaluate.py
└── tests/
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd "Multimodal Hate Detection via ZeroFew-Shot VLMs"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the example:

```bash
copy .env.example .env
```

Edit `.env` and add your API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Basic Usage

```python
from src.pipeline.detector import HateDetector

# Initialize detector
detector = HateDetector()

# Analyze a meme
result = await detector.detect(
    image_path="data/samples/test_meme.jpg",
    mode="zero_shot"  # or "few_shot", "cot"
)

print(f"Label: {result.classification.label}")
print(f"Justification: {result.classification.justification}")
```

### 4. Run Evaluation

```bash
# Evaluate on MultiBully dataset
python scripts/evaluate.py --dataset multibully --mode all

# Evaluate specific mode
python scripts/evaluate.py --dataset bangla --mode cot
```

## 📊 Supported Datasets

| Dataset | Language | Size | Labels |
|---------|----------|------|--------|
| MultiBully | Hindi-English | 5,854 | Bully/Non-bully, Sentiment, Sarcasm |
| BHM | Bangla/Code-mixed | 7,148 | Hateful/Non-hateful, Target |
| MUTE | Bangla/Code-mixed | 4,158 | Hateful/Non-hateful |
| BanglaAbuseMeme | Bangla | 4,043 | Abusive/Non-abusive |

## 🔬 Inference Modes

### Zero-Shot
- No examples provided
- Pure task description
- Tests model's inherent understanding

### Few-Shot
- 3-5 curated examples
- Covers diverse hate patterns
- Includes code-mixed examples

### Chain-of-Thought (CoT)
- Step-by-step reasoning
- Internal deliberation
- More robust for subtle cases

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Per-category breakdown
- Failure mode analysis

## 🔑 API Requirements

This project uses [OpenRouter](https://openrouter.ai/) for API access to:
- **VLM**: GPT-4o, Gemini 1.5 Pro, Claude 3
- **LLM**: GPT-4o, Claude 3, Llama 3

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.

## 📚 References

- MultiBully (SIGIR 2022) - Maity et al.
- Bengali Hateful Memes - Karim et al.
- MUTE Dataset - ACL Anthology
