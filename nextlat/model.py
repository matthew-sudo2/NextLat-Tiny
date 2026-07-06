"""
GPT-style transformer with RoPE embeddings and speculative proposal methods.
"""
import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Positional Embeddings
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        seq = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", seq, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
    
    def forward(self, x):
        # x: (batch, heads, seq_len, dim) or (batch, seq_len, heads, dim)
        if len(x.shape) == 4:
            seq_len = x.shape[2]
            cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        else:
            seq_len = x.shape[-2]
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]
        
        x_rot = torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
        return x * cos + x_rot * sin


# =============================================================================
# Attention and Feedforward Blocks
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
    
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q)
        k = self.rope(k)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(
            torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device),
            float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# Main GPT Model
# =============================================================================

class GPT(nn.Module):
    """Main transformer model with speculative proposal methods."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.output_head.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        """Forward pass through the model."""
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits
    
    def get_hidden_states(self, idx):
        """Get final layer hidden states."""
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.max_seq_len:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
        return idx
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    # =========================================================================
    # Speculative Decoding Methods (NextLat)
    # =========================================================================
    
    @torch.inference_mode()
    def speculative_propose(
        self,
        seq: torch.Tensor,                 # (batch, seq_len)
        latent_dynamics,                   # LatentDynamics model (p_psi)
        steps_to_propose: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate draft tokens using the latent dynamics model.

        Args:
            seq: Input sequence (batch, seq_len)
            latent_dynamics: The LatentDynamics model (p_psi)
            steps_to_propose: Number of tokens to draft
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)

        Returns:
            drafted_tokens: (batch, steps) tensor of drafted token IDs
            q_probs_steps: List of (batch, vocab_size) probability distributions for each step
        """
        # Get hidden states from the transformer
        hidden_states = self.get_hidden_states(seq)
        state = hidden_states[:, -1, :]  # (batch, hidden_size)

        drafted: List[torch.Tensor] = []
        q_probs_steps: List[torch.Tensor] = []

        for _ in range(steps_to_propose):
            # 1. Get logits from current state
            logits = self.output_head(state)  # (batch, vocab_size)

            # 2. Normalize with temperature and top-k
            logits = logits / temperature
            if top_k is not None:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, [-1]]] = -float('Inf')

            # 3. Sample a token
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            drafted.append(tok)
            q_probs_steps.append(probs)

            # 4. Predict next latent state using dynamics model
            # latent_dynamics expects (h_t, x_{t+1}) where x_{t+1} is token index
            state = latent_dynamics(state, tok.squeeze(1))  # (batch, hidden_size)

        return torch.cat(drafted, dim=1), q_probs_steps

    @torch.inference_mode()
    def target_probs_for_draft(
        self,
        seq: torch.Tensor,                 # (batch, seq_len)
        draft_tokens: torch.Tensor,        # (batch, draft_len)
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute target probabilities for verifying drafted tokens.

        Args:
            seq: Original sequence before drafting (batch, seq_len)
            draft_tokens: Drafted tokens (batch, draft_len)
            temperature: Sampling temperature
            top_k: Top-k for probability normalization

        Returns:
            p_probs_steps: (batch, draft_len, vocab_size) probabilities for each drafted position
            p_next_probs: (batch, vocab_size) probabilities for the next token after draft
        """
        # Concatenate seq + draft tokens
        full = torch.cat([seq, draft_tokens], dim=1)
        logits = self(full)  # (batch, seq_len + draft_len, vocab_size)

        prefix_len = seq.size(1)
        steps = draft_tokens.size(1)

        # Logits at positions corresponding to drafted tokens
        step_logits = logits[:, prefix_len - 1 : prefix_len - 1 + steps, :]  # (batch, steps, vocab)
        # Logits for the token after the last draft
        next_logits = logits[:, prefix_len - 1 + steps, :]  # (batch, vocab)

        # Normalize each step
        p_probs_steps = []
        for t in range(steps):
            logits_t = step_logits[:, t, :] / temperature
            if top_k is not None:
                top_k_vals, _ = torch.topk(logits_t, min(top_k, logits_t.size(-1)))
                logits_t[logits_t < top_k_vals[:, [-1]]] = -float('Inf')
            p_probs_steps.append(F.softmax(logits_t, dim=-1))

        # Normalize next token
        next_logits = next_logits / temperature
        if top_k is not None:
            top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < top_k_vals[:, [-1]]] = -float('Inf')
        p_next_probs = F.softmax(next_logits, dim=-1)

        return torch.stack(p_probs_steps, dim=1), p_next_probs