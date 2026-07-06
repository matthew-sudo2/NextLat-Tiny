# debug_nextlat_losses.py
import torch
import yaml
from tokenizers import Tokenizer
from nextlat.config import ModelConfig
from nextlat.model import GPT
from nextlat.latent_dynamics import LatentDynamics
from nextlat.losses import NextLatLoss
from data.prepare import TinyStoriesDataset
from torch.utils.data import DataLoader

def debug_loss_components():
    print("="*60)
    print("DEBUGGING NEXTLAT LOSS COMPONENTS")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    
    # Load config
    with open("configs/nextlat_d1_4096.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = ModelConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.vocab_size = vocab_size
    config.device = device
    
    # Load dataset
    dataset = TinyStoriesDataset(
        data_dir="data/tokenized",
        max_length=64
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    batch, targets = next(iter(dataloader))
    batch = batch.to(device)
    targets = targets.to(device)
    
    # Create model
    model = GPT(config)
    model.to(device)
    
    # Load your best NextLat checkpoint if it exists
    try:
        checkpoint = torch.load("checkpoints/nextlat_d1_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint!")
    except:
        print("No checkpoint found, using random weights")
    
    latent_dynamics = LatentDynamics(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        mlp_dim=config.latent_mlp_dim,
        num_layers=config.latent_mlp_layers
    )
    latent_dynamics.to(device)
    
    # Create loss function
    loss_fn = NextLatLoss(model, latent_dynamics, config)
    
    # Compute loss with detailed logging
    print("\nComputing loss...")
    model.eval()
    with torch.no_grad():
        loss, loss_dict = loss_fn.compute_loss(batch, targets)
    
    print("\nLoss Components:")
    print(f"  Total Loss: {loss.item():.4f}")
    print(f"  Next-Token Loss: {loss_dict['loss_next']:.4f}")
    print(f"  Next-H (MSE) Loss: {loss_dict['loss_next_h']:.4f}")
    print(f"  KL Loss: {loss_dict['loss_kl']:.4f}")
    
    # Calculate ratio
    total = loss.item()
    ntp_ratio = loss_dict['loss_next'] / total
    mse_ratio = loss_dict['loss_next_h'] / total
    kl_ratio = loss_dict['loss_kl'] / total
    
    print(f"\nLoss Ratios:")
    print(f"  NTP: {ntp_ratio:.2%}")
    print(f"  MSE: {mse_ratio:.2%}")
    print(f"  KL: {kl_ratio:.2%}")
    
    if mse_ratio > 0.5:
        print("\nMSE loss is dominating! Reduce lambda_next_h")
    if kl_ratio > 0.3:
        print("\nKL loss is too high! Reduce lambda_kl")
    if ntp_ratio < 0.3:
        print("\nNTP loss is too low! NextLat is ignoring next-token prediction")
    
    return loss_dict

if __name__ == "__main__":
    debug_loss_components()