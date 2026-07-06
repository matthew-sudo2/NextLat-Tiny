"""
Train a custom BPE tokenizer on TinyStories with proper UTF-8 handling
"""
import os
import re
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
from tqdm import tqdm

def clean_text(text):
    """Clean text to ensure valid UTF-8"""
    # Remove control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Replace non-UTF-8 characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text

def train_tokenizer(vocab_size=4096, save_path="data/tokenizer.json", num_samples=50000):
    """Train a BPE tokenizer directly from HuggingFace dataset"""
    
    print(f"Training tokenizer with vocab_size={vocab_size}")
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    # Use subset for tokenizer training
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Processing {len(dataset)} stories...")
    
    # Create temporary file with clean UTF-8 text
    os.makedirs("data", exist_ok=True)
    temp_file = "data/temp_texts.txt"
    
    print("Writing texts to temporary file...")
    with open(temp_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset):
            text = example["text"]
            # Clean text
            text = clean_text(text)
            # Write with proper newline
            f.write(text + "\n")
    
    # Initialize tokenizer with BPE model
    print("Initializing tokenizer...")
    tokenizer = Tokenizer(BPE(
        unk_token="[UNK]",
        fuse_unk=True,
    ))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Trainer with proper settings - NOW USING vocab_size parameter
    trainer = BpeTrainer(
        vocab_size=vocab_size,  # ← This was hardcoded before!
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=2,
        show_progress=True,
        continuing_subword_prefix="##",
        end_of_word_suffix="</w>",
    )
    
    # Train on the temporary file
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    try:
        tokenizer.train([temp_file], trainer)
    except Exception as e:
        print(f"Error during training: {e}")
        # Try alternative pre-tokenizer
        print("Retrying with ByteLevel pre-tokenizer...")
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.train([temp_file], trainer)
    
    # Add BOS/EOS processing
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    
    # Enable padding
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    
    # Save
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return tokenizer

def test_tokenizer(tokenizer_path="data/tokenizer.json"):
    """Test the tokenizer on sample text"""
    from tokenizers import Tokenizer
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        return
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    test_texts = [
        "Once upon a time, there was a little girl.",
        "The cat sat on the mat.",
        "She went to the store to buy some milk.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
        "12345 67890"
    ]
    
    print("\nTesting tokenizer:")
    print("-" * 50)
    
    for text in test_texts:
        try:
            encoded = tokenizer.encode(text)
            print(f"\nText: {text[:50]}...")
            print(f"Tokens: {encoded.tokens[:10]}...")
            print(f"Token IDs: {encoded.ids[:10]}...")
            print(f"Length: {len(encoded.ids)}")
        except Exception as e:
            print(f"Error encoding: {e}")

def save_tokenizer_info(tokenizer_path="data/tokenizer.json"):
    """Save tokenizer info for reference"""
    from tokenizers import Tokenizer
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        return
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    info_file = "data/tokenizer_info.txt"
    with open(info_file, "w", encoding="utf-8") as f:
        f.write("Tokenizer Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Vocabulary size: {tokenizer.get_vocab_size()}\n")
        f.write(f"Padding token: [PAD] (id: {tokenizer.token_to_id('[PAD]')})\n")
        f.write(f"Unknown token: [UNK] (id: {tokenizer.token_to_id('[UNK]')})\n")
        f.write(f"BOS token: [BOS] (id: {tokenizer.token_to_id('[BOS]')})\n")
        f.write(f"EOS token: [EOS] (id: {tokenizer.token_to_id('[EOS]')})\n")
        f.write("\nFirst 100 tokens:\n")
        for i in range(min(100, tokenizer.get_vocab_size())):
            token = tokenizer.id_to_token(i)
            f.write(f"  {i}: {token}\n")
    
    print(f"Tokenizer info saved to {info_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=4096, help="Vocabulary size")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples to use")
    args = parser.parse_args()
    
    print("=" * 50)
    print("TRAINING TOKENIZER")
    print("=" * 50)
    
    # Train tokenizer with specified vocab_size
    tokenizer = train_tokenizer(
        vocab_size=args.vocab_size,
        num_samples=args.num_samples
    )
    
    # Test it
    test_tokenizer()
    
    # Save info
    save_tokenizer_info()
    
    print("\n" + "=" * 50)
    print("TOKENIZER TRAINING COMPLETE!")
    print("=" * 50)