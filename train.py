"""
Main training script for NextLat
"""
import os
import sys
import argparse
import json
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(".")
from nextlat.config import ModelConfig, get_gpt_config, get_nextlat_config, get_mtp_config
from nextlat.model import GPT
from nextlat.latent_dynamics import LatentDynamics
from nextlat.losses import NextLatLoss
from nextlat.mtp_head import MTPHead, MTPModel
from data.prepare import TinyStoriesDataset

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_optimizer(model, config):
    """Get AdamW optimizer with weight decay"""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "ln" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    optimizer = optim.AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.95))
    return optimizer

def get_lr_scheduler(optimizer, config, total_steps):
    """Warmup + constant learning rate schedule"""
    def lr_lambda(step):
        if step < config.warmup_steps:
            return float(step) / float(max(1, config.warmup_steps))
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def train_step(model, loss_fn, batch, targets, optimizer, config):
    """Single training step"""
    batch = batch.to(config.device)
    targets = targets.to(config.device)
    
    # Forward pass
    loss, loss_dict = loss_fn.compute_loss(batch, targets)
    
    # Backward pass
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    return loss_dict

def train(config, model, loss_fn, optimizer, scheduler, train_loader, val_loader):
    """Main training loop"""
    device = config.device
    model.to(device)
    model.train()
    
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    pbar = tqdm(range(config.max_steps), desc="Training")
    for step in pbar:
        try:
            batch, targets = next(iter(train_loader))
        except StopIteration:
            train_loader = iter(train_loader)
            batch, targets = next(iter(train_loader))
        
        # Training step
        loss_dict = train_step(model, loss_fn, batch, targets, optimizer, config)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        if step % config.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "lr": f"{lr:.2e}",
                "step": step
            })
            
            # Log to file
            with open("results/train_log.txt", "a") as f:
                f.write(f"{step},{loss_dict['total_loss']:.4f},{lr:.2e}\n")