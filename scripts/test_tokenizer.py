"""
Test the trained tokenizer
"""
from tokenizers import Tokenizer
import os

def test_tokenizer():
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    
    print("Tokenizer loaded successfully!")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test encoding
    test_texts = [
        "Once upon a time, there was a little girl.",
        "The cat sat on the mat.",
        "She went to the store to buy some milk.",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
        "12345 67890"
    ]
    
    print("\nTesting tokenization:")
    print("-" * 60)
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"\nText: {text[:50]}...")
        print(f"Tokens: {encoded.tokens[:15]}...")
        print(f"Token IDs: {encoded.ids[:15]}...")
        print(f"Length: {len(encoded.ids)}")
        
        # Decode back
        decoded = tokenizer.decode(encoded.ids)
        print(f"Decoded: {decoded[:50]}...")

if __name__ == "__main__":
    test_tokenizer()