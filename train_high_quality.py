"""
High quality training with mixed precision disabled for stability
"""
import os
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from nextlat.config import ModelConfig
from nextlat.model import GPT
from nextlat.latent_dynamics import LatentDynamics
from nextlat.losses import NextLatLoss
from data.prepare import TinyStoriesDataset

def load_config(config_path):
    """Load and parse config with proper type conversion"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert string values to proper types
    for key, value in config_dict.items():
        if key == 'learning_rate' and isinstance(value, str):
            try:
                config_dict[key] = float(value)
            except ValueError:
                pass
        
        if isinstance(value, str):
            try:
                if '.' not in value:
                    config_dict[key] = int(value)
                else:
                    config_dict[key] = float(value)
            except ValueError:
                pass
    
    return config_dict

class CosineWarmupScheduler:
    """Cosine decay with warmup"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group.get('base_lr', 3e-4) * lr_scale
        
        return lr_scale

def train_high_quality_fixed(config_path=None):
    """High quality training with mixed precision disabled"""
    
    print("="*60)
    print("HIGH QUALITY TRAINING (FP32 - Stable)")
    print("="*60)
    
    # Load config
    if config_path and os.path.exists(config_path):
        config_dict = load_config(config_path)
        print(f"\nLoaded config from {config_path}")
    else:
        config_dict = {}
        print("\nNo config provided, using defaults")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Config
    config = ModelConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            if key == 'learning_rate':
                value = float(value)
            elif key in ['vocab_size', 'hidden_size', 'num_layers', 'num_heads', 
                        'max_seq_len', 'batch_size', 'grad_accum_steps', 
                        'warmup_steps', 'max_steps', 'eval_interval', 
                        'save_interval', 'log_interval', 'd', 'latent_mlp_layers']:
                value = int(value)
            setattr(config, key, value)
    
    config.device = device
    
    print(f"\nConfiguration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Seq length: {config.max_seq_len}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  d: {config.d}")
    print(f"  Precision: FP32 (no mixed precision)")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = TinyStoriesDataset(
        data_dir="data/tokenized",
        max_length=config.max_seq_len
    )
    print(f"Dataset size: {len(dataset):,} sequences")
    
    # Dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = GPT(config)
    model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create latent dynamics if d > 0
    latent_dynamics = None
    loss_fn = None
    
    if config.d > 0:
        print(f"Creating latent dynamics (d={config.d})...")
        latent_dynamics = LatentDynamics(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            mlp_dim=config.latent_mlp_dim,
            num_layers=config.latent_mlp_layers
        )
        latent_dynamics.to(device)
        print(f"Latent dynamics parameters: {latent_dynamics.count_parameters():,}")
        
        loss_fn = NextLatLoss(model, latent_dynamics, config)
    
    # Optimizer with base LR
    if latent_dynamics:
        optimizer = optim.AdamW(
            list(model.parameters()) + list(latent_dynamics.parameters()),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay
        )
    
    # Store base LR for scheduler
    for param_group in optimizer.param_groups:
        param_group['base_lr'] = config.learning_rate
    
    # Cosine scheduler with warmup
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.max_steps,
        min_lr_ratio=0.1
    )
    
    # Mixed precision DISABLED
    use_amp = False
    print("Using full precision (FP32)")
    
    # Training loop
    print("\nStarting training...")
    model.train()
    if latent_dynamics:
        latent_dynamics.train()
    
    step = 0
    running_loss = 0.0
    best_loss = float('inf')
    losses = []
    
    os.makedirs("checkpoints", exist_ok=True)
    
    model_name = "gpt"
    if config.d > 0:
        model_name = f"nextlat_d{config.d}"
    
    pbar = tqdm(range(config.max_steps), desc="Training")
    data_iter = iter(train_loader)
    
    for step in pbar:
        # Get batch
        try:
            batch, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch, targets = next(data_iter)
        
        batch = batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass (no AMP)
        if config.d > 0:
            loss, loss_dict = loss_fn.compute_loss(batch, targets)
        else:
            logits = model(batch)
            logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            targets_shifted = targets[:, 1:].reshape(-1)
            loss = nn.CrossEntropyLoss()(logits, targets_shifted)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if latent_dynamics:
            torch.nn.utils.clip_grad_norm_(latent_dynamics.parameters(), 1.0)
        optimizer.step()
        
        # Update scheduler
        lr_scale = scheduler.step()
        
        running_loss += loss.item()
        losses.append(loss.item())
        
        # Update progress bar
        if step % 10 == 0:
            avg_loss = running_loss / (step + 1)
            mem = f"{torch.cuda.memory_allocated()/1e9:.2f}GB" if device.type == "cuda" else "CPU"
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'lr': f"{scheduler.optimizer.param_groups[0]['lr']:.2e}",
                'mem': mem
            })
        
        # Clear cache periodically
        if step % 100 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save checkpoint
        if step % config.eval_interval == 0 and step > 0:
            avg_loss = running_loss / (step + 1)
            
            checkpoint_path = f"checkpoints/{model_name}_step_{step}.pt"
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses[-1000:],
                'config': config
            }
            
            if latent_dynamics:
                checkpoint['latent_dynamics_state_dict'] = latent_dynamics.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            print(f"\nStep {step}: Loss = {avg_loss:.4f}, LR = {scheduler.optimizer.param_groups[0]['lr']:.2e}")
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = f"checkpoints/{model_name}_best.pt"
                torch.save(checkpoint, best_path)
                print(f"New best model saved to {best_path} (loss: {avg_loss:.4f})")
    
    # Save final model
    avg_loss = running_loss / config.max_steps
    final_path = f"checkpoints/{model_name}_final.pt"
    checkpoint = {
        'step': config.max_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'losses': losses,
        'config': config
    }
    
    if latent_dynamics:
        checkpoint['latent_dynamics_state_dict'] = latent_dynamics.state_dict()
    
    torch.save(checkpoint, final_path)
    print(f"\nSaved final model to {final_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    train_high_quality_fixed(args.config)