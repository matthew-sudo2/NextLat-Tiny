"""
NextLat loss functions (Equations 3, 4, 5 from the paper)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

def smooth_l1_loss(pred, target, beta=1.0):
    """
    Smooth L1 loss (Huber loss)
    
    Args:
        pred: predicted values
        target: target values
        beta: threshold for switching between L1 and L2
    
    Returns:
        scalar loss value
    """
    diff = pred - target
    abs_diff = torch.abs(diff)
    cond = abs_diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, abs_diff - 0.5 * beta)
    return loss.mean()

def compute_next_token_loss(logits, targets, mask=None):
    """
    Compute next-token prediction loss (cross-entropy)
    
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        mask: (batch, seq_len) optional mask for prompt tokens
    
    Returns:
        scalar loss value
    """
    # Shift for next-token prediction
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()
    
    if mask is not None:
        mask = mask[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none"
        )
        loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)
    else:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
    
    return loss

def compute_next_h_loss(pred_h, target_h, mask=None):
    """
    Compute next-hidden state prediction loss (Smooth L1)
    Equation 3: L_next-h
    
    Args:
        pred_h: (batch, seq_len, hidden_size) predicted hidden states
        target_h: (batch, seq_len, hidden_size) target hidden states (with stop-grad)
        mask: (batch, seq_len) optional mask
    
    Returns:
        scalar loss value
    """
    if mask is not None:
        loss = smooth_l1_loss(pred_h, target_h)
        # Apply mask (simplified)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    else:
        loss = smooth_l1_loss(pred_h, target_h)
    
    return loss

def compute_kl_loss(pred_h, target_h, output_head, mask=None):
    """
    Compute KL divergence loss for token prediction alignment
    Equation 4: L_KL
    
    Uses a frozen/detached copy of the output head for target logits
    
    Args:
        pred_h: (batch, seq_len, hidden_size) predicted hidden states
        target_h: (batch, seq_len, hidden_size) target hidden states
        output_head: language model head (with stop-grad)
        mask: (batch, seq_len) optional mask
    
    Returns:
        scalar loss value
    """
    # Detach target (stop-gradient on posterior as per Equation 4)
    with torch.no_grad():
        # Get target logits from true hidden states
        target_logits = output_head(target_h)  # (batch, seq_len, vocab_size)
        target_probs = F.softmax(target_logits, dim=-1)
    
    # Get predicted logits (gradients flow here)
    pred_logits = output_head(pred_h)
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
    
    # Compute KL divergence
    kl = (target_probs * (target_probs.log() - pred_log_probs)).sum(dim=-1)
    
    if mask is not None:
        kl = (kl * mask).sum() / (mask.sum() + 1e-8)
    else:
        kl = kl.mean()
    
    return kl

class NextLatLoss:
    """
    Combined NextLat loss (Equation 5)
    
    L = L_next-token + lambda_next-h * L_next-h + lambda_KL * L_KL
    """
    
    def __init__(self, model, latent_dynamics, config):
        """
        Args:
            model: GPT model
            latent_dynamics: LatentDynamics model
            config: ModelConfig with d, next_h_lambda, kl_lambda
        """
        self.model = model
        self.latent_dynamics = latent_dynamics
        self.config = config
        
        self.next_h_lambda = config.next_h_lambda
        self.kl_lambda = config.kl_lambda
        self.d = config.d
        
        # Create frozen output head for KL loss (stop-gradient)
        # This is a deep copy that will not receive gradients
        self.frozen_output_head = deepcopy(model.output_head)
        for param in self.frozen_output_head.parameters():
            param.requires_grad = False
    
    def compute_loss(self, batch, targets, mask=None):
        """
        Compute full NextLat loss
        
        Args:
            batch: (batch, seq_len) input tokens
            targets: (batch, seq_len) target tokens
            mask: (batch, seq_len) optional mask for prompt tokens
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual loss components
        """
        batch_size, seq_len = batch.shape
        device = batch.device
        
        # Get hidden states and logits from transformer
        hidden_states = self.model.get_hidden_states(batch)  # (B, T, D)
        logits = self.model.output_head(hidden_states)  # (B, T, V)
        
        # 1. Next-token prediction loss
        loss_next = compute_next_token_loss(logits, targets, mask)
        
        # 2. Next-latent prediction loss (multi-step)
        loss_next_h = torch.tensor(0.0, device=device)
        loss_kl = torch.tensor(0.0, device=device)
        
        if self.d > 0:
            # Prepare for multi-step rollouts
            # Use teacher forcing with actual tokens
            current_h = hidden_states[:, :-1, :]  # h_t
            next_tokens = batch[:, 1:]  # x_{t+1}
            next_h_targets = hidden_states[:, 1:, :]  # h_{t+1} (targets)
            next_logits = logits[:, 1:, :]  # logits for h_{t+1}
            
            # Mask for valid positions
            if mask is not None:
                step_mask = mask[:, 1:]
            else:
                step_mask = None
            
            # Multi-step unrolling
            total_steps = min(self.d, seq_len - 1)
            
            # Initialize prediction with current hidden state
            pred_h = current_h[:, 0:1, :]  # Start with h_t at position 0
            
            for step in range(total_steps):
                # Get current hidden state (either from previous prediction or actual)
                if step == 0:
                    h_t = current_h[:, step, :]  # Use actual h_t for first step
                else:
                    # Use predicted hidden state from previous step
                    h_t = pred_h[:, -1, :]
                
                # Get next token for teacher forcing
                x_t = next_tokens[:, step]
                
                # Predict next hidden state using latent dynamics
                pred_next = self.latent_dynamics(h_t, x_t)
                
                # Target is actual next hidden state (with stop-gradient)
                target_next = next_h_targets[:, step, :].detach()
                
                # Mask for this step
                if step_mask is not None:
                    step_mask_i = step_mask[:, step]
                else:
                    step_mask_i = None
                
                # L_next-h (Equation 3)
                loss_next_h += compute_next_h_loss(
                    pred_next.unsqueeze(1), 
                    target_next.unsqueeze(1), 
                    step_mask_i.unsqueeze(1) if step_mask_i is not None else None
                )
                
                # L_KL (Equation 4) - only if lambda > 0
                if self.kl_lambda > 0:
                    # Get target logits for this position (detached)
                    target_logits = next_logits[:, step, :].detach()
                    target_probs = F.softmax(target_logits, dim=-1)
                    
                    # Predicted logits
                    pred_logits = self.model.output_head(pred_next)
                    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
                    
                    # KL divergence
                    kl_step = (target_probs * (target_probs.log() - pred_log_probs)).sum(dim=-1)
                    
                    if step_mask_i is not None:
                        kl_step = (kl_step * step_mask_i).sum() / (step_mask_i.sum() + 1e-8)
                    else:
                        kl_step = kl_step.mean()
                    
                    loss_kl += kl_step
                
                # Update prediction for next step
                pred_h = torch.cat([pred_h, pred_next.unsqueeze(1)], dim=1)
            
            # Average over steps
            loss_next_h = loss_next_h / total_steps
            loss_kl = loss_kl / total_steps
        
        # 3. Total loss (Equation 5)
        total_loss = loss_next + self.next_h_lambda * loss_next_h + self.kl_lambda * loss_kl
        
        # Return loss and dictionary of components
        loss_dict = {
            "loss_next": loss_next.item(),
            "loss_next_h": loss_next_h.item() if self.d > 0 else 0.0,
            "loss_kl": loss_kl.item() if self.d > 0 else 0.0,
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict

class MTPLoss:
    """
    Multi-Token Prediction loss (baseline)
    """
    
    def __init__(self, model, mtp_head, config):
        self.model = model
        self.mtp_head = mtp_head
        self.config = config
        self.mtp_lambda = getattr(config, 'mtp_lambda', 1.0)
        self.d = getattr(config, 'mtp_d', 2)
    
    def compute_loss(self, batch, targets, mask=None):
        """Compute MTP loss"""
        batch_size, seq_len = batch.shape
        device = batch.device
        
        # Get hidden states
        hidden_states = self.model.get_hidden_states(batch)
        logits = self.model.output_head(hidden_states)
        
        # Base next-token loss
        loss_next = compute_next_token_loss(logits, targets, mask)
        
        # MTP predictions
        # For simplicity, we just predict multiple tokens with separate heads
        # This is a simplified version - full MTP uses transformer blocks
        loss_mtp = torch.tensor(0.0, device=device)
        
        for i in range(1, self.d + 1):
            # Predict token at offset i+1
            offset = i + 1
            if offset < seq_len:
                # Use hidden state at position t to predict token at t+offset
                h_for_pred = hidden_states[:, :-offset, :]
                pred_logits = self.model.output_head(h_for_pred)
                
                # Target is token at offset
                target_tokens = targets[:, offset:].contiguous()
                
                # Only use valid positions
                valid_len = min(pred_logits.size(1), target_tokens.size(1))
                if valid_len > 0:
                    loss_i = F.cross_entropy(
                        pred_logits[:, :valid_len, :].reshape(-1, pred_logits.size(-1)),
                        target_tokens[:, :valid_len].reshape(-1)
                    )
                    loss_mtp += loss_i
        
        loss_mtp = loss_mtp / self.d if self.d > 0 else 0.0
        total_loss = loss_next + self.mtp_lambda * loss_mtp
        
        loss_dict = {
            "loss_next": loss_next.item(),
            "loss_mtp": loss_mtp.item() if self.d > 0 else 0.0,
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_dict

def compute_nextlat_loss(model, latent_dynamics, batch, targets, config, mask=None):
    """
    Convenience function for computing NextLat loss
    
    Args:
        model: GPT model
        latent_dynamics: LatentDynamics model
        batch: input tokens
        targets: target tokens
        config: ModelConfig
        mask: optional mask
    
    Returns:
        total_loss, loss_dict
    """
    loss_fn = NextLatLoss(model, latent_dynamics, config)
    return loss_fn.compute_loss(batch, targets, mask)