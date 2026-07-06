"""
Self-speculative decoding implementation for NextLat.
Implements the draft-then-verify loop from Leviathan et al. 2022.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import time

def normalize_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Normalize logits with temperature and optional top-k/top-p filtering.
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    if top_k is not None and top_k > 0:
        # Top-k filtering
        top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_k_vals[:, [-1]]] = -float('Inf')
    
    if top_p is not None and top_p < 1.0:
        # Top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float('Inf'))
    
    return F.softmax(logits, dim=-1)

def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Sample a token from probability distribution."""
    return torch.multinomial(probs, num_samples=1)

def speculative_decode(
    model,                      # GPT model with speculative_propose and target_probs_for_draft
    latent_dynamics,            # LatentDynamics model
    prompt: torch.Tensor,       # (batch, prompt_len)
    max_new_tokens: int,
    draft_length: int = 5,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Self-speculative decoding loop.

    Args:
        model: GPT model with speculative_propose and target_probs_for_draft
        latent_dynamics: LatentDynamics model (p_psi)
        prompt: Input sequence (batch, prompt_len)
        max_new_tokens: Maximum tokens to generate
        draft_length: Number of tokens to draft per step (gamma)
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p (nucleus) filtering

    Returns:
        Generated sequence (batch, prompt_len + generated_tokens)
    """
    seq = prompt
    total_generated = 0
    
    while total_generated < max_new_tokens:
        # Determine how many tokens to draft (don't exceed remaining)
        current_draft_len = min(draft_length, max_new_tokens - total_generated)
        if current_draft_len <= 0:
            break
        
        # 1. Draft phase
        draft_tokens, q_probs = model.speculative_propose(
            seq, latent_dynamics, current_draft_len, temperature, top_k
        )
        
        # 2. Verify phase
        p_probs_steps, p_next_probs = model.target_probs_for_draft(
            seq, draft_tokens, temperature, top_k
        )
        
        # 3. Acceptance/rejection loop
        n_accepted = 0
        for i in range(current_draft_len):
            # Random number for acceptance
            r = torch.rand(seq.size(0), device=seq.device)
            
            # Get probabilities for the drafted token
            q_prob = q_probs[i].gather(1, draft_tokens[:, i:i+1]).squeeze(1)
            p_prob = p_probs_steps[:, i, :].gather(1, draft_tokens[:, i:i+1]).squeeze(1)
            
            # Acceptance probability: min(1, p_target / p_q)
            accept_prob = torch.min(torch.ones_like(p_prob), p_prob / (q_prob + 1e-8))
            
            # Accept if random < accept_prob
            accept = r < accept_prob
            if accept.all():
                n_accepted += 1
            else:
                # Reject: sample from adjusted distribution
                # For simplicity, we stop at the first rejection
                break
        
        # 4. Append accepted tokens
        if n_accepted > 0:
            seq = torch.cat([seq, draft_tokens[:, :n_accepted]], dim=1)
            total_generated += n_accepted
        
        # 5. If no tokens accepted or all accepted, sample from target distribution
        if n_accepted == 0:
            # Sample from the target distribution for the next token
            next_token = torch.multinomial(p_next_probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)
            total_generated += 1
        elif n_accepted == current_draft_len:
            # All drafted tokens accepted, sample one more token from target
            next_token = torch.multinomial(p_next_probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)
            total_generated += 1
    
    return seq

def measure_speedup(
    model,
    latent_dynamics,
    prompt: torch.Tensor,
    max_new_tokens: int = 100,
    draft_lengths: List[int] = [2, 3, 4, 5, 6, 8, 10],
    temperature: float = 0.8,
    num_runs: int = 5,
) -> dict:
    """
    Measure speedup of speculative decoding vs autoregressive.

    Returns:
        Dict with speedup results for each draft length.
    """
    results = {}
    
    # Check if CUDA is available for synchronization
    cuda_available = torch.cuda.is_available()
    
    # Autoregressive baseline
    print("Measuring autoregressive baseline...")
    ar_times = []
    for _ in range(num_runs):
        if cuda_available:
            torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            _ = model.generate(prompt, max_new_tokens, temperature=temperature)
        
        if cuda_available:
            torch.cuda.synchronize()
        ar_times.append(time.time() - start)
    
    ar_avg_time = sum(ar_times) / len(ar_times)
    ar_tokens_per_sec = max_new_tokens / ar_avg_time
    
    print(f"Autoregressive: {ar_avg_time:.3f}s, {ar_tokens_per_sec:.1f} tok/s")
    print("\nMeasuring speculative decoding...")
    
    for draft_len in draft_lengths:
        print(f"  Draft length {draft_len}...")
        spec_times = []
        accepted_counts = []
        
        for _ in range(num_runs):
            if cuda_available:
                torch.cuda.synchronize()
            start = time.time()
            
            generated = speculative_decode(
                model, latent_dynamics, prompt,
                max_new_tokens=max_new_tokens,
                draft_length=draft_len,
                temperature=temperature
            )
            
            if cuda_available:
                torch.cuda.synchronize()
            spec_times.append(time.time() - start)
            
            # Count how many tokens were actually generated
            tokens_generated = generated.size(1) - prompt.size(1)
            accepted_counts.append(tokens_generated)
        
        avg_time = sum(spec_times) / len(spec_times)
        avg_accepted = sum(accepted_counts) / len(accepted_counts)
        speedup = ar_avg_time / avg_time
        
        results[draft_len] = {
            'avg_time': avg_time,
            'avg_accepted': avg_accepted,
            'speedup': speedup,
            'tokens_per_sec': avg_accepted / avg_time,
        }
        
        print(f"    Speedup: {speedup:.2f}x, Accepted: {avg_accepted:.1f} tokens")
    
    return {
        'autoregressive': {
            'avg_time': ar_avg_time,
            'tokens_per_sec': ar_tokens_per_sec,
        },
        'speculative': results,
    }