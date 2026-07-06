"""
run_evaluation.py - Comprehensive evaluation of all trained models
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer
from tqdm import tqdm
import argparse
import json
from tabulate import tabulate

from nextlat.config import ModelConfig
from nextlat.model import GPT
from data.prepare import TinyStoriesDataset


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get('config', None)
    if ckpt_config is None:
        config = ModelConfig()
        config.vocab_size = 4096
        config.hidden_size = 384
        config.num_layers = 6
        config.num_heads = 6
        config.max_seq_len = 256
    else:
        config = ModelConfig()
        for key in ['vocab_size', 'hidden_size', 'num_layers', 'num_heads', 
                    'max_seq_len', 'dropout', 'd', 'next_h_lambda', 'kl_lambda',
                    'latent_mlp_dim', 'latent_mlp_layers']:
            if hasattr(ckpt_config, key):
                setattr(config, key, getattr(ckpt_config, key))
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    config.vocab_size = tokenizer.get_vocab_size()
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config, checkpoint.get('loss', None)

def compute_perplexity(model, dataloader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for i, (batch, targets) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if max_batches and i >= max_batches:
                break
            batch = batch.to(device)
            targets = targets.to(device)
            logits = model(batch)
            logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            targets_shifted = targets[:, 1:].reshape(-1)
            loss = criterion(logits, targets_shifted)
            total_loss += loss.item()
            total_tokens += targets_shifted.numel()
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def load_models(checkpoint_dir="checkpoints", device="cuda"):
    models = {}
    checkpoints = {
        "GPT": "gpt_final.pt",
        "NextLat d=1": "nextlat_d1_final.pt",
        "NextLat d=2": "nextlat_d2_final.pt",
    }
    for name, ckpt_file in checkpoints.items():
        path = os.path.join(checkpoint_dir, ckpt_file)
        if os.path.exists(path):
            try:
                model, config, loss = load_model_from_checkpoint(path, device)
                models[name] = {
                    'model': model,
                    'config': config,
                    'checkpoint_loss': loss,
                    'path': path
                }
                print(f"Loaded {name} from {path} (loss: {loss:.4f})")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        else:
            print(f"Checkpoint not found: {path}")
    return models

def print_table(results):
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    table_data = []
    for name, metrics in results.items():
        table_data.append([
            name,
            f"{metrics['loss']:.4f}",
            f"{metrics['perplexity']:.2f}",
            f"{metrics['params']:,}",
            f"{metrics['checkpoint_loss']:.4f}" if metrics['checkpoint_loss'] else "N/A"
        ])
    headers = ["Model", "Validation Loss", "Perplexity", "Parameters", "Checkpoint Loss"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--eval_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    print("\nLoading validation dataset...")
    dataset = TinyStoriesDataset(
        data_dir="data/tokenized",
        max_length=args.max_seq_len,
        split="validation"
    )
    if args.eval_samples and args.eval_samples < len(dataset):
        dataset = Subset(dataset, range(min(args.eval_samples, len(dataset))))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    print(f"Validation samples: {len(dataset)}, Batches: {len(dataloader)}")
    
    print("\nLoading models...")
    models = load_models(args.checkpoint_dir, device)
    if not models:
        print("No models found. Exiting.")
        return
    
    results = {}
    print("\nEvaluating models...")
    for name, info in models.items():
        print(f"\n{name}:")
        model = info['model']
        avg_loss, perplexity = compute_perplexity(model, dataloader, device)
        results[name] = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'checkpoint_loss': info['checkpoint_loss'],
            'params': model.count_parameters()
        }
        print(f"  Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    print_table(results)
    
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/evaluation_results.json")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if "GPT" in results:
        baseline_loss = results["GPT"]['loss']
        print(f"\nBaseline: GPT (Loss: {baseline_loss:.4f})")
        print("\nComparison to baseline:")
        for name, metrics in results.items():
            if name != "GPT":
                gap = metrics['loss'] - baseline_loss
                print(f"  {name}: +{gap:.4f} ({gap/baseline_loss*100:+.2f}%)")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()