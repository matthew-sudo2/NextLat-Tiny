"""
Tokenize TinyStories and save as sharded binary files
"""
import os
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
import torch

def tokenize_dataset(
    text_file="data/raw/tinystories_train.txt",
    tokenizer_path="data/tokenizer.json",
    output_dir="data/tokenized",
    max_length=256,
    shard_size=1000000,  # tokens per shard
):
    """Tokenize and shard the dataset"""
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Read all texts
    with open(text_file, "r") as f:
        texts = f.read().split("\n\n")
    
    token_ids = []
    shard_id = 0
    
    for text in tqdm(texts, desc="Tokenizing"):
        if not text.strip():
            continue
            
        # Tokenize
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        
        # Add to buffer
        token_ids.extend(ids)
        
        # Save shard when buffer is large enough
        while len(token_ids) >= shard_size:
            shard = token_ids[:shard_size]
            token_ids = token_ids[shard_size:]
            
            # Save as numpy array
            np.save(f"{output_dir}/shard_{shard_id:05d}.npy", np.array(shard, dtype=np.uint16))
            shard_id += 1
    
    # Save remaining
    if token_ids:
        np.save(f"{output_dir}/shard_{shard_id:05d}.npy", np.array(token_ids, dtype=np.uint16))
    
    print(f"Saved {shard_id + 1} shards to {output_dir}")

class TinyStoriesDataset:
    """PyTorch dataset for TinyStories"""
    
    def __init__(self, data_dir="data/tokenized", max_length=256, split="train"):
        self.data_dir = data_dir
        self.max_length = max_length
        self.shards = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
        
        # Build shard index
        self.shard_offsets = []
        self.total_tokens = 0
        
        for shard in self.shards:
            arr = np.load(f"{data_dir}/{shard}")
            self.shard_offsets.append(self.total_tokens)
            self.total_tokens += len(arr)
        
        print(f"Loaded {len(self.shards)} shards, {self.total_tokens:,} total tokens")
    
    def __len__(self):
        return self.total_tokens // self.max_length
    
    def __getitem__(self, idx):
        start_token = idx * self.max_length
        end_token = start_token + self.max_length + 1  # +1 for next token prediction
        
        # Find which shard contains this token
        shard_idx = np.searchsorted(self.shard_offsets, start_token, side="right") - 1
        shard_offset = self.shard_offsets[shard_idx]
        
        # Load shard
        arr = np.load(f"{self.data_dir}/{self.shards[shard_idx]}")
        
        # Get sequence
        local_start = start_token - shard_offset
        local_end = end_token - shard_offset
        sequence = arr[local_start:local_end]
        
        # Pad if necessary
        if len(sequence) < self.max_length + 1:
            sequence = np.pad(sequence, (0, self.max_length + 1 - len(sequence)), constant_values=0)
        
        # Return input and target
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        
        return x, y

if __name__ == "__main__":
    tokenize_dataset()