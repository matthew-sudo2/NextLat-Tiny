"""
Test speculative decoding with trained NextLat model.
"""
import os
import torch
from tokenizers import Tokenizer
from nextlat.config import ModelConfig
from nextlat.model import GPT
from nextlat.latent_dynamics import LatentDynamics
from speculative_sampling import speculative_decode, measure_speedup

def test_speculative_decoding():
    print("="*60)
    print("TESTING SPECULATIVE DECODING WITH TRAINED NEXTLAT")
    print("="*60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")
    
    # ====== LOAD TRAINED NEXTLAT CHECKPOINT ======
    checkpoint_path = "checkpoints/nextlat_d1_final.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    ckpt_config = checkpoint['config']
    print(f"Checkpoint step: {checkpoint['step']}")
    print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    
    # Create model using config from checkpoint
    config = ModelConfig()
    # Copy relevant attributes from checkpoint config
    for key in ['vocab_size', 'hidden_size', 'num_layers', 'num_heads', 
                'max_seq_len', 'dropout', 'd', 'next_h_lambda', 'kl_lambda',
                'latent_mlp_dim', 'latent_mlp_layers']:
        if hasattr(ckpt_config, key):
            setattr(config, key, getattr(ckpt_config, key))
    
    # Override vocab_size with tokenizer's (just in case)
    config.vocab_size = vocab_size
    
    print(f"\nModel config from checkpoint:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  d: {config.d}")
    
    # Load GPT backbone
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Load latent dynamics
    latent_dynamics = LatentDynamics(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        mlp_dim=config.latent_mlp_dim,
        num_layers=config.latent_mlp_layers
    )
    if 'latent_dynamics_state_dict' in checkpoint:
        latent_dynamics.load_state_dict(checkpoint['latent_dynamics_state_dict'])
        print("✅ Loaded latent dynamics from checkpoint")
    else:
        print("⚠️ No latent dynamics found, using random weights")
    
    latent_dynamics.to(device)
    latent_dynamics.eval()
    print(f"Latent dynamics parameters: {latent_dynamics.count_parameters():,}")
    
    # Prepare prompt
    prompt_text = "Once upon a time, there was a little"
    print(f"\nPrompt: '{prompt_text}'")
    
    encoded = tokenizer.encode(prompt_text)
    prompt = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    print(f"Prompt tokens: {prompt.shape}")
    
    # Generate with autoregressive
    print("\n" + "="*60)
    print("AUTOREGRESSIVE GENERATION")
    print("="*60)
    
    with torch.no_grad():
        ar_output = model.generate(prompt, max_new_tokens=30, temperature=0.8)
    ar_text = tokenizer.decode(ar_output[0].cpu().tolist())
    print(f"\n{ar_text}")
    
    # Generate with speculative decoding
    print("\n" + "="*60)
    print("SPECULATIVE DECODING (draft_length=5)")
    print("="*60)
    
    spec_output = speculative_decode(
        model, latent_dynamics, prompt,
        max_new_tokens=30,
        draft_length=5,
        temperature=0.8
    )
    spec_text = tokenizer.decode(spec_output[0].cpu().tolist())
    print(f"\n{spec_text}")
    
    # Speed comparison
    print("\n" + "="*60)
    print("SPEED COMPARISON (50 tokens)")
    print("="*60)
    
    try:
        results = measure_speedup(
            model, latent_dynamics, prompt,
            max_new_tokens=50,
            draft_lengths=[2, 4, 6, 8, 10],
            num_runs=3
        )
        
        print("\n" + "="*60)
        print("SPEEDUP RESULTS")
        print("="*60)
        print(f"\nAutoregressive: {results['autoregressive']['tokens_per_sec']:.1f} tok/s")
        print("\nDraft Length | Speedup | Accepted/Step")
        print("-" * 40)
        for draft_len, spec in results['speculative'].items():
            accepted_per_step = spec['avg_accepted'] / (50 / draft_len) if draft_len > 0 else 0
            print(f"     {draft_len:2d}      |  {spec['speedup']:.2f}x   |    {accepted_per_step:.2f}")
            
            # Key claim: accepted tokens per step > draft length
            if accepted_per_step > draft_len:
                print(f"    ✅ Accepted ({accepted_per_step:.1f}) > d ({draft_len})")
            else:
                print(f"    ⚠️ Accepted ({accepted_per_step:.1f}) ≤ d ({draft_len})")
                
    except Exception as e:
        print(f"Speed test failed: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_speculative_decoding()