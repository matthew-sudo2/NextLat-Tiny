"""
Download TinyStories dataset from Hugging Face
"""
import os
from datasets import load_dataset
import json

def download_tinystories(split="train", num_samples=None):
    """Download TinyStories dataset"""
    dataset = load_dataset("roneneldan/TinyStories")
    
    if split == "train":
        data = dataset["train"]
    else:
        data = dataset["validation"]
    
    if num_samples:
        data = data.select(range(min(num_samples, len(data))))
    
    # Save as text file for tokenizer training
    os.makedirs("data/raw", exist_ok=True)
    
    texts = [example["text"] for example in data]
    with open(f"data/raw/tinystories_{split}.txt", "w") as f:
        for text in texts:
            f.write(text + "\n\n")
    
    print(f"Downloaded {len(texts)} {split} samples")
    return texts

if __name__ == "__main__":
    download_tinystories("train")
    download_tinystories("validation")