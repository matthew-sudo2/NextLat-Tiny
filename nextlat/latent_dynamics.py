"""
Latent dynamics model p_psi (Equation 9 from paper)
Predicts next latent state: h_{t+1} = f_psi(h_t, x_{t+1}) + h_t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentDynamics(nn.Module):
    """
    Predicts next latent state: h_{t+1} = f_psi(h_t, x_{t+1}) + h_t
    
    Input: (h_t, x_{t+1}) 
    Output: h_{t+1} (predicted)
    
    This is a 3-layer MLP with residual connection as described in the paper.
    """
    
    def __init__(self, hidden_size, vocab_size, mlp_dim=None, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        
        if mlp_dim is None:
            mlp_dim = 2 * hidden_size
        
        # Embedding for tokens
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # MLP: takes concat of h_t and embed(x_{t+1})
        input_dim = 2 * hidden_size
        
        layers = []
        layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, mlp_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(mlp_dim, mlp_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(mlp_dim, hidden_size))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, h_t, x_t_plus_1):
        """
        Predict next hidden state
        
        Args:
            h_t: (batch, hidden_size) or (batch, seq_len, hidden_size)
                Current hidden state
            x_t_plus_1: (batch,) or (batch, seq_len)
                Next token indices
        
        Returns:
            h_{t+1}: same shape as h_t
                Predicted next hidden state
        """
        # Embed token
        token_embed = self.token_embedding(x_t_plus_1)
        
        # Handle different input shapes
        if len(h_t.shape) == 2:
            # Batch case: (batch, hidden_size)
            concat = torch.cat([h_t, token_embed], dim=-1)
            delta = self.mlp(concat)
            h_next = h_t + delta
        else:
            # Sequence case: (batch, seq_len, hidden_size)
            concat = torch.cat([h_t, token_embed], dim=-1)
            delta = self.mlp(concat)
            h_next = h_t + delta
        
        return h_next
    
    def rollout(self, h_0, x_sequence, steps):
        """
        Rollout the dynamics model for multiple steps
        
        Args:
            h_0: (batch, hidden_size) initial hidden state
            x_sequence: (batch, steps) tokens to condition on
            steps: number of steps to rollout
        
        Returns:
            List of predicted hidden states
        """
        h = h_0
        predictions = []
        
        for step in range(steps):
            x_t = x_sequence[:, step]
            h = self(h, x_t)
            predictions.append(h)
        
        return torch.stack(predictions, dim=1)  # (batch, steps, hidden_size)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters())

class LatentDynamicsSimple(nn.Module):
    """
    Simplified latent dynamics for testing
    """
    
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Simple embedding and linear projection
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        
        # Initialize
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, h_t, x_t_plus_1):
        token_embed = self.token_embedding(x_t_plus_1)
        concat = torch.cat([h_t, token_embed], dim=-1)
        delta = self.fc(concat)
        delta = self.ln(delta)
        return h_t + delta