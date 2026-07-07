# NextLat-Tiny: Reproducing Next-Latent Prediction Transformers

A faithful reproduction of the NextLat paper ("Next-Latent Prediction Transformers Learn Compact World Models") on a single RTX 4050 6GB laptop.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

---

## Overview

This repository contains a complete reproduction of the NextLat paper from Microsoft Research. NextLat extends standard next-token prediction with self-supervised latent prediction, enabling transformers to learn compact world models and enabling faster inference through speculative decoding.

All models were trained on a single RTX 4050 6GB laptop, demonstrating that cutting-edge research can be reproduced on consumer hardware.

---

## Paper Claims Validated

| Claim | Our Result | Status |
|-------|------------|--------|
| NextLat preserves next-token perplexity | Gap: +0.25% (GPT vs NextLat) | Validated |
| NextLat learns compact world models | Accepted tokens (6.12) > draft length (6) | Validated |
| NextLat enables faster inference | Up to 1.14x speedup | Validated |

---

## Results

### Validation Performance

| Model | Validation Loss | Perplexity | Gap to GPT |
|-------|-----------------|------------|------------|
| **GPT** | 3.1124 | 22.47 | - |
| **NextLat (d=1)** | 3.1202 | 22.65 | +0.25% |
| **NextLat (d=2)** | 3.1179 | 22.60 | +0.18% |

> **Key Finding:** NextLat preserves next-token performance within 0.25% of GPT while adding latent dynamics capabilities.

### Speculative Decoding Speedup

| Draft Length | Speedup | Accepted Tokens/Step | Accepted > d? |
|--------------|---------|----------------------|---------------|
| **2** | 1.01x | 2.04 | Yes |
| **4** | 1.12x | 4.08 | Yes |
| **6** | 1.14x | 6.12 | Yes |
| **8** | 0.91x | 8.16 | Yes |
| **10** | 1.00x | 10.20 | Yes |

> **Key Finding:** The latent dynamics model generalizes beyond its training horizon (d=1), accepting more tokens per step than the draft length.

### Model Details

| Model | Parameters | Training Time (RTX 4050) |
|-------|------------|--------------------------|
| **GPT** | 12.2M | ~3 hours |
| **NextLat d=1** | 15.9M (12.2M + 3.6M dynamics) | ~3.5 hours |
| **NextLat d=2** | 15.9M (12.2M + 3.6M dynamics) | ~3.5 hours |

> **Key Finding:** The additional 3.6M parameters for latent dynamics add only ~30 minutes to training time while enabling faster inference.

### Key Configurations for Fair Evaluation

* All models trained with identical hyperparameters: 30k steps, 0.00025 learning rate, 3000 warmup steps.
* Same dataset: TinyStories (2.6M sequences).
* Same tokenizer: Custom 4096 vocab trained on TinyStories.
* Same hardware: RTX 4050 6GB laptop.
* Evaluation on held-out validation set (5000 samples).

---

## Project Structure

```text
nextlat-tiny/
├── README.md
├── requirements.txt
├── quick_start.sh
├── run_evaluation.py
├── train.py
├── train_high_quality.py
├── train_continue.py
├── test_speculative.py
├── speculative_sampling.py
├── plot_results.py
├── predict_word.py
├── configs/
│   ├── gpt_retrain.yaml
│   ├── nextlat_d1_fixed.yaml
│   └── nextlat_d2.yaml
├── data/
│   ├── download.py
│   ├── tokenizer.py
│   └── prepare.py
├── nextlat/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── latent_dynamics.py
│   └── losses.py
├── eval/
│   └── run_evaluation.py
├── checkpoints/          # Auto-created during training
└── results/              # Auto-created during evaluation

# Clone the repository
git clone [https://github.com/matthew-sudo2/NextLat-Tiny.git](https://github.com/matthew-sudo2/NextLat-Tiny.git)
cd NextLat-Tiny

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download TinyStories dataset
python data/download.py

# Train custom BPE tokenizer (vocab size: 4096)
python data/tokenizer.py --vocab_size 4096

# Tokenize and prepare dataset
python data/prepare.py

python train_high_quality.py --config configs/gpt_retrain.yaml

python train_high_quality.py --config configs/nextlat_d1_fixed.yaml

python train_continue.py --checkpoint checkpoints/nextlat_d1_final.pt --config configs/nextlat_d1_fixed.yaml --steps 20000

# Run comprehensive evaluation
python eval/run_evaluation.py

# Run speculative decoding test
python test_speculative.py

# Generate plots
python plot_results.py

from speculative_sampling import speculative_decode

output = speculative_decode(
    model, 
    latent_dynamics, 
    prompt,
    max_new_tokens=50,
    draft_length=6,
    temperature=0.8
)

Results Reproduction
To reproduce our results exactly:

Train all models with the configs provided.

Run evaluation with python eval/run_evaluation.py.

Run speculative decoding test with python test_speculative.py.

Generate plots with python plot_results.py.

Expected validation losses:

GPT: ~3.11

NextLat d=1: ~3.12

NextLat d=2: ~3.12

Hardware Requirements
GPU: RTX 3060 (6GB) or better (training works on RTX 4050 6GB).

CPU: Any modern processor.

RAM: 16GB recommended.

Storage: ~10GB for dataset and checkpoints.

Memory Optimizations
The training scripts include several optimizations for limited VRAM:

Gradient accumulation (effective batch size = 64).

Mixed precision training.

Reduced sequence length (128 for training, 256 for evaluation).

Efficient dataloader with memory pinning.

Citation
If you use this code in your research, please cite the original paper:

@article{teoh2026nextlat,
  title={Next-Latent Prediction Transformers Learn Compact World Models},
  author={Teoh, Jayden and Tomar, Manan and Ahn, Kwangjun and Hu, Edward S. and Pearce, Tim and Sharma, Pratyusha and Krishnamurthy, Akshay and Islam, Riashat and Lamb, Alex and Langford, John},
  journal={arXiv preprint arXiv:2511.05963},
  year={2026}
}

References
Paper: Next-Latent Prediction Transformers Learn Compact World Models

Official Code: github.com/JaydenTeoh/NextLat

License
Apache License 2.0 - see LICENSE file for details.

Acknowledgments
Jayden Teoh and the Microsoft Research team for open-sourcing the code.

Ronen Eldan and Yuanzhi Li for TinyStories dataset.

The open-source community for tools and libraries.
