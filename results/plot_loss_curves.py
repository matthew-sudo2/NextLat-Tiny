"""
plot_loss_curves.py - Plot loss curves for GPT, NextLat d=1, NextLat d=2
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
        except:
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

def load_best_loss(checkpoint_dir="checkpoints", filename="*_best.pt"):
    """Load best loss from a model."""
    files = glob(os.path.join(checkpoint_dir, filename))
    if files:
        try:
            ckpt = torch.load(files[0], map_location="cpu", weights_only=False)
            return ckpt.get('loss', None)
        except:
            pass
    return None

def plot_loss_curves():
    """Plot training loss curves for all models."""
    
    # Define model patterns
    models = {
        "GPT": {
            "step_pattern": "gpt_step_*.pt",
            "final": "gpt_final.pt",
            "best": "gpt_best.pt",
            "color": "#2ecc71",
            "label": "GPT",
            "marker": "o",
            "linestyle": "-"
        },
        "NextLat d=1": {
            "step_pattern": "nextlat_d1_step_*.pt",
            "final": "nextlat_d1_final.pt",
            "best": "nextlat_d1_best.pt",
            "color": "#3498db",
            "label": "NextLat d=1",
            "marker": "s",
            "linestyle": "-"
        },
        "NextLat d=2": {
            "step_pattern": "nextlat_d2_step_*.pt",
            "final": "nextlat_d2_final.pt",
            "best": "nextlat_d2_best.pt",
            "color": "#e74c3c",
            "label": "NextLat d=2",
            "marker": "^",
            "linestyle": "-"
        }
    }
    
    # Check if there are any checkpoints
    has_data = False
    for name, config in models.items():
        files = glob(os.path.join("checkpoints", config["step_pattern"]))
        if files:
            has_data = True
            break
    
    if not has_data:
        print("No step checkpoints found. Please run training first.")
        return
    
    # Create figure with clean style
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style similar to reference image
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    all_losses = {}
    
    for name, config in models.items():
        # Load step checkpoints
        losses = load_checkpoint_losses("checkpoints", config["step_pattern"])
        
        # Add final loss if available
        final_loss = load_final_loss("checkpoints", config["final"])
        if final_loss is not None:
            max_step = max(losses.keys()) if losses else 0
            losses[max_step + 100] = final_loss
        
        if losses:
            steps = sorted(losses.keys())
            loss_vals = [losses[s] for s in steps]
            
            # Plot line with markers
            ax.plot(steps, loss_vals, 
                   marker=config["marker"],
                   markersize=5,
                   markevery=max(1, len(steps)//10),  # Show every 10th marker
                   linewidth=2.5,
                   color=config["color"],
                   label=config["label"],
                   linestyle=config["linestyle"],
                   alpha=0.9)
            
            all_losses[name] = (steps, loss_vals)
    
    # Add final loss annotations
    y_min = float('inf')
    y_max = float('-inf')
    for name, (steps, loss_vals) in all_losses.items():
        final_step = steps[-1]
        final_loss = loss_vals[-1]
        if final_loss < y_min:
            y_min = final_loss
        if final_loss > y_max:
            y_max = final_loss
        
        # Annotate final loss
        ax.annotate(f'{final_loss:.4f}', 
                   xy=(final_step, final_loss),
                   xytext=(5, -15),
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold',
                   color=config["color"])
    
    # Add horizontal grid lines (like reference image)
    ax.grid(True, linestyle='-', alpha=0.3, color='#cccccc')
    
    # Set labels and title
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Curves: GPT vs NextLat', fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    
    # Add gap annotation
    if len(all_losses) >= 2:
        final_losses = {name: losses[-1] for name, (_, losses) in all_losses.items()}
        gpt_loss = final_losses.get("GPT", None)
        if gpt_loss is not None and "NextLat d=1" in final_losses:
            gap1 = final_losses["NextLat d=1"] - gpt_loss
            gap2 = final_losses["NextLat d=2"] - gpt_loss if "NextLat d=2" in final_losses else 0
            ax.text(0.02, 0.95, f'Gap to GPT: +{max(gap1, gap2):.4f}', 
                   transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))
    
    # Set y-axis to start at 0 or slightly above min loss
    y_min = max(0, y_min - 0.5)
    y_max = y_max + 0.5
    ax.set_ylim(y_min, y_max)
    
    # Set x-axis limits
    max_step = max([max(steps) for steps, _ in all_losses.values()])
    ax.set_xlim(0, max_step + 500)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/loss_curves.png", dpi=300, bbox_inches="tight", facecolor='white')
    print("Plot saved to results/loss_curves.png")
    
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
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(models))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Bar chart for loss
    bars1 = ax1.bar(x, losses, color=colors, edgecolor='black', linewidth=1, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.set_ylabel('Validation Loss (lower is better)')
    ax1.set_title('Validation Loss Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars1, losses):
        ax1.text(bar.get_x() + bar.get_width()/2., v + 0.01, f'{v:.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Bar chart for perplexity
    bars2 = ax2.bar(x, perplexities, color=colors, edgecolor='black', linewidth=1, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.set_ylabel('Perplexity (lower is better)')
    ax2.set_title('Perplexity Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars2, perplexities):
        ax2.text(bar.get_x() + bar.get_width()/2., v + 0.5, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("results/validation_comparison.png", dpi=200, bbox_inches="tight", facecolor='white')
    print("Validation comparison plot saved to results/validation_comparison.png")
    plt.show()

def main():
    print("="*60)
    print("PLOTTING LOSS CURVES")
    print("="*60)
    
    plot_loss_curves()
    plot_validation_comparison()
    
    print("\nPlots saved to results/")
    print("  - loss_curves.png")
    print("  - validation_comparison.png")

if __name__ == "__main__":
    main()