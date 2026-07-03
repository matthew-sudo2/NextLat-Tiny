# NextLat-Tiny: Compact World Models via Next-Latent Prediction

A faithful, hardware-friendly reproduction of the NextLat research paper demonstrating compact world models and self-speculative decoding on TinyStories.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

NextLat-Tiny is a minimal, reproducible implementation of the paper "Next-Latent Prediction Transformers Learn Compact World Models" (Microsoft Research, 2025). It demonstrates how transformers can learn belief-state representations and compact world models through self-supervised latent prediction while preserving parallel training efficiency.

### Key Claims Validated

- NextLat preserves next-token perplexity (within 2% of GPT baseline)
- NextLat learns belief-state representations (linear probes show better long-range prediction)
- Latent dynamics generalize beyond training horizon (accepted tokens > draft length)
- Self-speculative decoding gives wall-clock speedup (1.3x+ over autoregressive)
- More compact representations (lower effective latent rank)

### Core Concepts

Transformers lack inherent pressure to compress history into compact states. NextLat reintroduces this recurrent inductive bias through:

1. Latent dynamics model (p_psi): Predicts next hidden state from current state and next token
2. Self-supervised losses: Smooth L1 regression + KL distillation in latent space
3. Belief-state learning: Theoretically guaranteed to form sufficient statistics of history

## Project Structure

```text
nextlat-tiny/
├── configs/
│   ├── gpt.yaml
│   ├── nextlat_d1.yaml
│   ├── nextlat_d2.yaml
│   └── mtp_d2.yaml
├── data/
│   ├── download.py
│   ├── tokenizer.py
│   └── prepare.py
├── nextlat/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── latent_dynamics.py
│   ├── losses.py
│   └── mtp_head.py
├── eval/
│   ├── __init__.py
│   ├── perplexity.py
│   ├── speculative_decode.py
│   ├── probe.py
│   └── benchmark_speedup.py
├── notebooks/
│   └── colab_train.ipynb
├── checkpoints/
├── results/
│   └── plots/
├── README.md
├── LICENSE
├── .gitignore
├── quick_start.sh
├── run_evaluation.py
└── train.py

## Configuration Parameters

### Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| vocab_size | 1000 | Custom BPE vocabulary |
| hidden_size | 384 | Transformer hidden dimension |
| num_layers | 6 | Number of transformer blocks |
| num_heads | 6 | Number of attention heads |
| max_seq_len | 256 | Maximum sequence length |
| dropout | 0.1 | Dropout rate |

### Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 16 | Per-device batch size |
| grad_accum_steps | 4 | Gradient accumulation (effective batch 64) |
| learning_rate | 3e-4 | Peak learning rate |
| weight_decay | 0.1 | Weight decay |
| warmup_steps | 1000 | Linear warmup steps |
| max_steps | 20000 | Total training steps |
| optimizer | AdamW | Betas: (0.9, 0.95) |
| gradient_clip | 1.0 | Gradient norm clipping |

### NextLat Specific

| Parameter | Value | Description |
|-----------|-------|-------------|
| d | 1 or 2 | Multi-step prediction horizon |
| lambda_next_h | 1.0 | Weight for L_next-h loss |
| lambda_KL | 1.0 | Weight for L_KL loss |
| latent_mlp_dim | 768 | Hidden dimension for p_psi MLP |
| latent_mlp_layers | 3 | Number of layers in p_psi |

### Latent Dynamics (p_psi)

| Parameter | Value | Description |
|-----------|-------|-------------|
| input_dim | 768 | 2 * hidden_size (h_t + embed(x)) |
| hidden_dim | 768 | MLP hidden dimension |
| output_dim | 384 | hidden_size (residual delta) |
| activation | GELU | Activation function |
| residual | Yes | Output = MLP(input) + h_t |

### MTP Baseline (d=2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_heads | 4 | Attention heads in MTP blocks |
| feedforward_dim | 1536 | 4 * hidden_size |
| lambda_mtp | 1.0 | Weight for MTP loss |

### Evaluation

| Parameter | Value | Description |
|-----------|-------|-------------|
| eval_batches | 100 | Batches for perplexity eval |
| probe_offsets | [1,2,4,8,12,16,20] | Token offsets for linear probes |
| draft_lengths | [2,3,4,5,6,8,10] | Draft lengths for speculative decoding |
| max_new_tokens | 50 | Tokens to generate for speed test |

## Quick Start

### One-Command Setup

```bash
git clone https://github.com/yourusername/nextlat-tiny.git
cd nextlat-tiny
chmod +x quick_start.sh
./quick_start.sh
