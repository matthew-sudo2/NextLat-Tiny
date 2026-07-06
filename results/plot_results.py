"""
plot_results.py - Plot validation results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_validation_results():
    """Plot validation loss and perplexity from JSON results."""
    
    # Load results
    if not os.path.exists("results/evaluation_results.json"):
        print("Run eval/run_evaluation.py first to generate results.")
        return
    
    with open("results/evaluation_results.json", "r") as f:
        results = json.load(f)
    
    models = list(results.keys())
    losses = [results[m]['loss'] for m in models]
    perplexities = [results[m]['perplexity'] for m in models]
    
    # Colors matching your models
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    
    # Loss plot
    bars1 = ax1.bar(models, losses, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.set_ylabel('Validation Loss (lower is better)', fontsize=12)
    ax1.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, v in zip(bars1, losses):
        ax1.text(bar.get_x() + bar.get_width()/2., v + 0.01, f'{v:.4f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Perplexity plot
    bars2 = ax2.bar(models, perplexities, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax2.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax2.set_title('Perplexity Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, v in zip(bars2, perplexities):
        ax2.text(bar.get_x() + bar.get_width()/2., v + 0.3, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/validation_plot.png", dpi=200, bbox_inches="tight", facecolor='white')
    print("✅ Plot saved to results/validation_plot.png")
    
    plt.show()

def plot_training_curves():
    """Plot training loss curves from checkpoints."""
    import torch
    from glob import glob
    
    models = {
        "GPT": {"pattern": "gpt_step_*.pt", "color": "#2ecc71", "label": "GPT"},
        "NextLat d=1": {"pattern": "nextlat_d1_step_*.pt", "color": "#3498db", "label": "NextLat d=1"},
        "NextLat d=2": {"pattern": "nextlat_d2_step_*.pt", "color": "#e74c3c", "label": "NextLat d=2"}
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    has_data = False
    
    for name, config in models.items():
        files = sorted(glob(f"checkpoints/{config['pattern']}"))
        if not files:
            continue
        
        steps = []
        losses = []
        for f in files:
            try:
                ckpt = torch.load(f, map_location="cpu", weights_only=False)
                step = ckpt.get('step', 0)
                loss = ckpt.get('loss', None)
                if loss is not None and step > 0:
                    steps.append(step)
                    losses.append(loss)
            except:
                pass
        
        if steps:
            has_data = True
            ax.plot(steps, losses, 
                   marker='o', markersize=5, markevery=max(1, len(steps)//8),
                   linewidth=2.5, color=config["color"], label=config["label"])
            
            # Annotate final point
            ax.annotate(f'{losses[-1]:.4f}', 
                       xy=(steps[-1], losses[-1]),
                       xytext=(5, -10), textcoords='offset points',
                       fontsize=10, fontweight='bold', color=config["color"])
    
    if not has_data:
        print("No step checkpoints found. Skipping training curves.")
        return
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=200, bbox_inches="tight", facecolor='white')
    print("✅ Training curves saved to results/training_curves.png")
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Always plot validation results
    plot_validation_results()
    
    # Try to plot training curves if checkpoints exist
    plot_training_curves()