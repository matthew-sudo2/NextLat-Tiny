"""
Multi-Token Prediction (MTP) baseline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPHead(nn.Module):
    """
    Multi-token prediction head for MTP baseline
    
    Similar to Gloeckle et al. 2024
    """
    
    def __init__(self, hidden_size, vocab_size, d=2):
        super().__init__()
        self.d = d
        
        # Additional transformer blocks for each prediction depth
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=4 * hidden_size,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(d)
        ])
        
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(d)
        ])
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states, token_embeddings):
        """
        Predict multiple future tokens
        
        Args:
            hidden_states: (batch, seq_len, hidden_size) from transformer
            token_embeddings: (batch, seq_len, hidden_size) token embeddings
        
        Returns:
            List of (batch, seq_len, vocab_size) predictions for each offset
        """
        predictions = []
        
        # Shift inputs
        h = hidden_states[:, :-1, :]  # h_t
        x_emb = token_embeddings[:, 1:, :]  # x_{t+1}
        
        for i in range(self.d):
            # Use previous predictions as input
            if i == 0:
                h_input = h
            else:
                # Combine with predicted token embeddings
                h_input = h + x_emb
            
            # Apply transformer block
            h_output = self.blocks[i](h_input, h_input)
            
            # Predict token
            logits = self.output_heads[i](h_output)
            predictions.append(logits)
            
            # Update for next step
            if i < self.d - 1:
                # Get predicted token embeddings
                pred_tokens = logits.argmax(dim=-1)
                x_emb = token_embeddings[:, 2+i:, :]  # Shift further
        
        return predictions

class MTPModel:
    """Wrapper for MTP training"""
    
    def __init__(self, base_model, mtp_head, config):
        self.base_model = base_model
        self.mtp_head = mtp_head
        self.config = config
        
        # Weight for MTP loss
        self.lambda_mtp = 1.0
    
    def compute_loss(self, batch, targets, mask=None):
        """Compute MTP loss"""
        # Get hidden states
        hidden_states = self.base_model.get_hidden_states(batch)
        logits = self.base_model.output_head(hidden_states)
        
        # Base next-token loss
        loss_next = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1)
        )
        
        # MTP predictions
        token_embeddings = self.base_model.token_embedding(batch)
        mtp_logits = self.mtp_head(hidden_states, token_embeddings)
        
        # MTP losses
        loss_mtp = 0.0
        for i, logits_i in enumerate(mtp_logits):
            offset = i + 2  # predict token at t+i+2
            if offset < targets.size(1):
                loss_mtp += F.cross_entropy(
                    logits_i.reshape(-1, logits_i.size(-1)),
                    targets[:, offset:].reshape(-1)[:logits_i.numel() // logits_i.size(-1)]
                )
        
        loss_mtp = loss_mtp / len(mtp_logits)
        
        total_loss = loss_next + self.lambda_mtp * loss_mtp
        
        return total_loss, {
            "loss_next": loss_next.item(),
            "loss_mtp": loss_mtp.item(),
            "total_loss": total_loss.item()
        }