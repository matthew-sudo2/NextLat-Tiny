"""
predict_word.py - Interactive script that predicts the next word
"""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer
from nextlat.config import ModelConfig
from nextlat.model import GPT
from nextlat.latent_dynamics import LatentDynamics


class WordPredictor:
    def __init__(self, checkpoint_path="checkpoints/gpt_final.pt", use_nextlat=False):
        """Initialize the word predictor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file("data/tokenizer.json")
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # Setup config
        self.config = ModelConfig()
        self.config.vocab_size = self.vocab_size
        self.config.hidden_size = 384
        self.config.num_layers = 6
        self.config.num_heads = 6
        self.config.max_seq_len = 256
        
        # Load model
        self.model = GPT(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Load latent dynamics if available and requested
        self.latent_dynamics = None
        if use_nextlat and 'latent_dynamics_state_dict' in checkpoint:
            self.latent_dynamics = LatentDynamics(
                self.config.hidden_size,
                self.config.vocab_size
            )
            self.latent_dynamics.load_state_dict(checkpoint['latent_dynamics_state_dict'])
            self.latent_dynamics.to(self.device)
            self.latent_dynamics.eval()
            print("Loaded latent dynamics for speculative prediction")
    
    def predict_next_word(self, prompt, temperature=0.8):
        """Predict the next word given a prompt."""
        # Encode prompt
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=self.device)
        
        # Get prediction
        with torch.no_grad():
            # Get logits for the next token
            logits = self.model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get the most likely token (greedy)
            best_token_id = torch.argmax(probs).item()
            best_token = self.tokenizer.decode([best_token_id])
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs, min(5, len(probs)))
            top_tokens = []
            for idx, prob in zip(top_indices, top_probs):
                token = self.tokenizer.decode([idx.item()])
                top_tokens.append((token, prob.item()))
        
        return best_token.strip(), top_tokens
    
    def run(self):
        """Run interactive prediction loop."""
        print("\n" + "="*60)
        print("WORD PREDICTOR")
        print("="*60)
        print("Type your prompt and I'll predict the next word.")
        print("Type 'exit' or 'quit' to stop.")
        print("Type 'spec' to toggle speculative decoding (if available).")
        print("="*60)
        
        use_speculative = False
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'spec':
                    use_speculative = not use_speculative
                    status = "ON" if use_speculative else "OFF"
                    print(f"Speculative mode: {status}")
                    continue
                
                if not prompt:
                    continue
                
                # Predict
                best_word, top_tokens = self.predict_next_word(prompt)
                
                print(f"\nPrompt: {prompt}")
                print(f"Next word: {best_word}")
                print("\nTop 5 predictions:")
                for i, (token, prob) in enumerate(top_tokens, 1):
                    bar = "█" * int(prob * 30)
                    print(f"  {i}. {token:10} ({prob:.2%}) {bar}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gpt_final.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--nextlat", action="store_true",
                       help="Use NextLat with speculative prediction")
    args = parser.parse_args()
    
    predictor = WordPredictor(args.checkpoint, args.nextlat)
    predictor.run()


if __name__ == "__main__":
    import argparse
    main()