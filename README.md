# neuron-hacking-guide

Code examples and research notes on language model representation editing techniques.

## Overview & Purpose
neuron-hacking-guide explores techniques for modifying specific factual associations within Transformer language models (ROME, MEMIT, SAE, FiNE) without retraining full weight matrices.

## Key Features
- Implementation of Rank-One Model Editing (ROME) algorithms.
- Sparse Autoencoder (SAE) feature extraction scripts.
- Model activation editing utilities.

## Tech Stack & Dependencies
- **Language**: Python 3.9+
- **Libraries**: PyTorch, HuggingFace Transformers, NNsight

## Project Structure
```text
neuron-hacking-guide/
├── experiments/
├── chapters/
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.9+
- PyTorch with CUDA support

### Steps
```bash
git clone https://github.com/zsomborturcsanyi7-lang/neuron-hacking-guide.git
cd neuron-hacking-guide
pip install -r requirements.txt
```

## Usage Examples
```bash
python experiments/run_rome.py --model gpt2-xl
```

## Status & License
Status: Research Notes & Experiments.
License: MIT
