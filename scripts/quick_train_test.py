"""
Quick training test to verify GPU setup
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from nextlat.config import ModelConfig
from nextlat.model import GPT
from data.prepare import TinyStoriesDataset

def quick_test():
    print("="*60)
    print("QUICK TRAINING TEST")
    print("="*60)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Config for quick test
    config = ModelConfig()
    config.vocab_size = 1000
    config.hidden_size = 256  # Smaller for quick test
    config.num_layers = 4     # Fewer layers
    config.num_heads = 4
    config.max_seq_len = 64   # Shorter sequences
    config.batch_size = 8
    config.learning_rate = 3e-4
    config.d = 0
    config.device = device
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = TinyStoriesDataset(
        data_dir="data/tokenized",
        max_length=config.max_seq_len
    )
    
    # Use only first 1000 sequences for quick test
    # Instead of dataset.data, use Subset
    from torch.utils.data import Subset
    subset_indices = list(range(min(1000, len(dataset))))
    dataset = Subset(dataset, subset_indices)
    print(f"Using {len(dataset)} sequences")
    
    # Dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    
    # Model
    print("\nCreating model...")
    model = GPT(config)
    model.to(device)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    model.train()
    print("\nTraining for 100 steps...")
    
    use_amp = device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision")
    
    losses = []
    
    # Create iterator
    data_iter = iter(train_loader)
    
    for step in range(100):
        try:
            batch, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch, targets = next(data_iter)
        
        batch = batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(batch)
                # Shift for next-token prediction
                logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
                targets_shifted = targets[:, 1:].reshape(-1)
                loss = nn.CrossEntropyLoss()(logits, targets_shifted)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            targets_shifted = targets[:, 1:].reshape(-1)
            loss = nn.CrossEntropyLoss()(logits, targets_shifted)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        losses.append(loss.item())
        
        if step % 10 == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:]) if losses else loss.item()
            mem = f"{torch.cuda.memory_allocated()/1e9:.2f}GB" if device.type == "cuda" else "CPU"
            print(f"Step {step}: loss={avg_loss:.4f}, mem={mem}")
    
    # Final results
    final_loss = sum(losses[-10:]) / len(losses[-10:]) if losses else 0
    print(f"\nFinal loss after 100 steps: {final_loss:.4f}")
    
    if device.type == "cuda":
        print(f"Peak memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE!")
    print(f"✅ Training works on {device}")
    print("="*60)
    
    return model

if __name__ == "__main__":
    quick_test()