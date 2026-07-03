# NextLat-Tiny: Compact World Models via Next-Latent Prediction

A faithful, hardware-friendly reproduction of the **NextLat** research paper, demonstrating compact world models and self-speculative decoding on the TinyStories dataset.

---

## Overview

**NextLat-Tiny** is a minimal, reproducible implementation of the paper **"Next-Latent Prediction Transformers Learn Compact World Models"** (Microsoft Research, 2025). It demonstrates how transformers can learn belief-state representations and compact world models through self-supervised latent prediction while preserving the parallel training efficiency of standard transformers.

## Key Claims Validated

* Preserves next-token perplexity (within **2%** of a GPT baseline)
* Learns belief-state representations (linear probes achieve better long-range prediction)
* Latent dynamics generalize beyond the training horizon (accepted tokens exceed draft length)
* Self-speculative decoding provides wall-clock speedups (**1.3×+** over autoregressive decoding)
* Produces more compact representations (lower effective latent rank)

## Core Concepts

Traditional transformers have little incentive to compress history into compact latent states. NextLat reintroduces this recurrent inductive bias through:

1. **Latent Dynamics Model (`pψ`)** – Predicts the next hidden state from the current hidden state and the next token.
2. **Self-Supervised Latent Losses** – Uses Smooth L1 regression together with KL distillation in latent space.
3. **Belief-State Learning** – Encourages hidden states to become sufficient statistics of the observed history.

---

# Project Structure

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
```

---

# Configuration Parameters

## Model Architecture

| Parameter     | Value | Description                  |
| ------------- | ----: | ---------------------------- |
| `vocab_size`  |  1000 | Custom BPE vocabulary size   |
| `hidden_size` |   384 | Transformer hidden dimension |
| `num_layers`  |     6 | Number of transformer layers |
| `num_heads`   |     6 | Number of attention heads    |
| `max_seq_len` |   256 | Maximum sequence length      |
| `dropout`     |   0.1 | Dropout probability          |

## Training

| Parameter          |  Value | Description                                       |
| ------------------ | -----: | ------------------------------------------------- |
| `batch_size`       |     16 | Per-device batch size                             |
| `grad_accum_steps` |      4 | Gradient accumulation (effective batch size = 64) |
| `learning_rate`    |   3e-4 | Peak learning rate                                |
| `weight_decay`     |    0.1 | Weight decay                                      |
| `warmup_steps`     |   1000 | Linear warmup steps                               |
| `max_steps`        | 20,000 | Total training steps                              |
| `optimizer`        |  AdamW | β=(0.9, 0.95)                                     |
| `gradient_clip`    |    1.0 | Gradient norm clipping                            |

## NextLat

| Parameter           |  Value | Description                      |
| ------------------- | -----: | -------------------------------- |
| `d`                 | 1 or 2 | Multi-step prediction horizon    |
| `lambda_next_h`     |    1.0 | Weight of **L<sub>next-h</sub>** |
| `lambda_KL`         |    1.0 | Weight of **L<sub>KL</sub>**     |
| `latent_mlp_dim`    |    768 | Hidden dimension of `pψ`         |
| `latent_mlp_layers` |      3 | Number of MLP layers             |

## Latent Dynamics (`pψ`)

| Parameter    | Value | Description                         |
| ------------ | ----: | ----------------------------------- |
| `input_dim`  |   768 | `2 × hidden_size` (`hₜ + embed(x)`) |
| `hidden_dim` |   768 | Hidden dimension                    |
| `output_dim` |   384 | Hidden state dimension              |
| `activation` |  GELU | Activation function                 |
| `residual`   |   Yes | Output = `MLP(input) + hₜ`          |

## MTP Baseline (`d = 2`)

| Parameter         | Value | Description                                |
| ----------------- | ----: | ------------------------------------------ |
| `num_heads`       |     4 | Attention heads                            |
| `feedforward_dim` |  1536 | Feed-forward dimension (`4 × hidden_size`) |
| `lambda_mtp`      |   1.0 | Weight of the MTP loss                     |

## Evaluation

| Parameter        |                Value | Description                                |
| ---------------- | -------------------: | ------------------------------------------ |
| `eval_batches`   |                  100 | Batches for perplexity evaluation          |
| `probe_offsets`  | `[1,2,4,8,12,16,20]` | Offsets used for linear probes             |
| `draft_lengths`  |   `[2,3,4,5,6,8,10]` | Draft lengths for speculative decoding     |
| `max_new_tokens` |                   50 | Tokens generated during speed benchmarking |

---

# Quick Start

## One-Command Setup

```bash
git clone https://github.com/yourusername/nextlat-tiny.git
cd nextlat-tiny
chmod +x quick_start.sh
./quick_start.sh
```
