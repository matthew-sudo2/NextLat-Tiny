"""
Verify the tokenized dataset
"""
import os
import numpy as np
import torch
from tokenizers import Tokenizer

def verify_shards():
    """Check the shards and dataset statistics"""
    
    print("="*60)
    print("VERIFYING TOKENIZED DATASET")
    print("="*60)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    print(f"\nTokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    # Check shards
    shard_dir = "data/tokenized"
    shards = sorted([f for f in os.listdir(shard_dir) if f.endswith(".npy")])
    print(f"\nFound {len(shards)} shards")
    
    # Analyze first few shards
    total_tokens = 0
    total_files = 0
    
    print("\nAnalyzing shards...")
    for i, shard_file in enumerate(shards[:5]):  # Check first 5 shards
        path = os.path.join(shard_dir, shard_file)
        data = np.load(path)
        total_tokens += len(data)
        total_files += 1
        
        print(f"  Shard {i+1} ({shard_file}): {len(data):,} tokens")
        print(f"    Token range: {data.min()} - {data.max()}")
        print(f"    Unique tokens: {len(np.unique(data))}")
        
        # Show first 20 tokens
        print(f"    First 20 tokens: {data[:20].tolist()}")
    
    # Estimate total tokens
    avg_shard_size = total_tokens / total_files
    estimated_total = avg_shard_size * len(shards)
    
    print(f"\nEstimated total tokens: {estimated_total:,.0f}")
    print(f"Estimated sequences (len=256): {estimated_total // 256:,}")
    
    return shards

def test_sample_sequence():
    """Test loading a sequence from the dataset"""
    print("\n" + "="*60)
    print("TESTING SEQUENCE LOADING")
    print("="*60)
    
    from data.prepare import TinyStoriesDataset
    
    # Create dataset with small sample
    print("\nCreating dataset (first 100 sequences)...")
    dataset = TinyStoriesDataset(
        data_dir="data/tokenized",
        max_length=256,
        split="train"
    )
    
    print(f"Dataset size: {len(dataset):,} sequences")
    
    # Load first 5 sequences
    print("\nLoading first 5 sequences:")
    for i in range(min(5, len(dataset))):
        x, y = dataset[i]
        print(f"\n  Sequence {i+1}:")
        print(f"    Input shape: {x.shape}")
        print(f"    Target shape: {y.shape}")
        print(f"    Non-padding tokens: {(x != 0).sum().item()}/{len(x)}")
        print(f"    First 20 input tokens: {x[:20].tolist()}")
        print(f"    First 20 target tokens: {y[:20].tolist()}")

if __name__ == "__main__":
    # Verify shards
    shards = verify_shards()
    
    # Test sequence loading
    test_sample_sequence()
    
    print("\n" + "="*60)
    print("DATASET VERIFICATION COMPLETE!")
    print("Ready for training!")
    print("="*60)