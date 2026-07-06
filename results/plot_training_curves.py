"""
plot_training_curves.py - Plot training loss curves for all models
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import json

def load_checkpoint_losses(checkpoint_dir="checkpoints", pattern="*_step_*.pt"):
    """Load losses from step checkpoints for a given model pattern."""
    losses = {}
    files = sorted(glob(os.path.join(checkpoint_dir, pattern)))
    
    for f in files:
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            step = ckpt.get('step', 0)
            loss = ckpt.get('loss', None)
            if loss is not None and step > 0:
                losses[step] = loss
        except Exception as e:
            pass
    
    return losses

def load_final_loss(checkpoint_dir="checkpoints", filename="*_final.pt"):
    """Load final loss from a model."""
    files = glob(os.path.join(checkpoint_dir, filename))
    if files:
        try:
            ckpt = torch.load(files[0], map_location="cpu", weights_only=False)
            return ckpt.get('loss', None)
        except:
            pass
    return None

def get_model_patterns():
    """Define checkpoint patterns for each model."""
    return {
        "GPT": {
            "step_pattern": "gpt_step_*.pt",
            "final": "gpt_final.pt",
            "color": "#2ecc71",
            "label": "GPT"
        },
        "NextLat d=1": {
            "step_pattern": "nextlat_d1_step_*.pt",
            "final": "nextlat_d1_final.pt",
            "color": "#3498db",
            "label": "NextLat d=1"
        },
        "NextLat d=2": {
            "step_pattern": "nextlat_d2_step_*.pt",
            "final": "nextlat_d2_final.pt",
            "color": "#e74c3c",
            "label": "NextLat d=2"
        }
    }

def plot_training_curves():
    """Plot training loss curves for all models."""
    
    models = get_model_patterns()
    
    # Check if there are any checkpoints
    has_data = False
    for name, config in models.items():
        files = glob(os.path.join("checkpoints", config["step_pattern"]))
        if files:
            has_data = True
            break
    
    if not has_data:
        print("No step checkpoints found. Please run training first.")
        print("Looking for patterns: gpt_step_*.pt, nextlat_d1_step_*.pt, nextlat_d2_step_*.pt")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_losses = {}
    
    for name, config in models.items():
        # Load step checkpoints
        losses = load_checkpoint_losses("checkpoints", config["step_pattern"])
        
        # Add final loss if available
        final_loss = load_final_loss("checkpoints", config["final"])
        if final_loss is not None:
            max_step = max(losses.keys()) if losses else 0
            losses[max_step + 1] = final_loss
        
        if losses:
            steps = sorted(losses.keys())
            loss_vals = [losses[s] for s in steps]
            
            ax.plot(steps, loss_vals, 
                   marker='o', 
                   markersize=4,
                   linewidth=2,
                   color=config["color"],
                   label=config["label"])
            
            all_losses[name] = (steps, loss_vals)
    
    if not all_losses:
        print("No checkpoint data found. Nothing to plot.")
        return
    
    # Add vertical line at max steps
    max_steps = max([max(steps) for steps, _ in all_losses.values()])
    ax.axvline(x=max_steps, color='gray', linestyle='--', alpha=0.5)
    ax.text(max_steps + 50, ax.get_ylim()[0] + 0.1, 
            f'Final Step: {max_steps}', 
            fontsize=10, color='gray')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss (lower is better)', fontsize=12)
    ax.set_title('Training Loss Curves: GPT vs NextLat', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for final losses
    for name, (steps, loss_vals) in all_losses.items():
        final_step = steps[-1]
        final_loss = loss_vals[-1]
        ax.annotate(f'{final_loss:.4f}', 
                   xy=(final_step, final_loss),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9,
                   fontweight='bold')
    
    # Highlight the small gap between models
    if len(all_losses) >= 2:
        final_losses = {name: losses[-1] for name, (_, losses) in all_losses.items()}
        gpt_loss = final_losses.get("GPT", None)
        if gpt_loss is not None:
            max_gap = max([v - gpt_loss for k, v in final_losses.items() if k != 'GPT'])
            ax.text(0.02, 0.02, f'Gap to GPT: +{max_gap:.4f}', transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_curves.png", dpi=200, bbox_inches="tight")
    print("Plot saved to results/training_curves.png")
    
    plt.show()

def plot_validation_comparison():
    """Plot validation loss comparison."""
    try:
        with open("results/evaluation_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Run eval/run_evaluation.py first to generate validation results.")
        return
    
    models = list(results.keys())
    losses = [results[m]['loss'] for m in models]
    perplexities = [results[m]['perplexity'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(models))
    
    # Line chart for loss
    ax1.plot(x, losses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.set_ylabel('Validation Loss (lower is better)')
    ax1.set_title('Validation Loss Comparison')
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(losses):
        ax1.annotate(f'{v:.4f}', xy=(i, v), xytext=(0, 5), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    # Line chart for perplexity
    ax2.plot(x, perplexities, 'rs-', linewidth=2, markersize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.set_ylabel('Perplexity (lower is better)')
    ax2.set_title('Perplexity Comparison')
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(perplexities):
        ax2.annotate(f'{v:.2f}', xy=(i, v), xytext=(0, 5), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("results/validation_comparison.png", dpi=200, bbox_inches="tight")
    print("Validation comparison plot saved to results/validation_comparison.png")
    plt.show()

def main():
    print("Generating plots...")
    plot_training_curves()
    plot_validation_comparison()
    print("Done!")

if __name__ == "__main__":
    main()