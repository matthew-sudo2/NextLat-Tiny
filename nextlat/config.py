"""
Configuration for NextLat models
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    """Main configuration class for all models"""
    
    # Model architecture
    vocab_size: int = 1000
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    max_seq_len: int = 256
    dropout: float = 0.1
    
    # Training
    batch_size: int = 16
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 20000
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 100
    
    # NextLat specific
    d: int = 0  # 0 = no latent prediction (GPT), 1, 2, etc.
    next_h_lambda: float = 1.0
    kl_lambda: float = 1.0
    latent_mlp_dim: int = 768
    latent_mlp_layers: int = 3
    
    # MTP specific
    mtp_d: int = 0
    mtp_lambda: float = 1.0
    
    # Other
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42
    
    def __post_init__(self):
        """Validate config after initialization"""
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"

# Helper functions for creating configs
def get_gpt_config(**kwargs):
    """Get GPT baseline config"""
    config = ModelConfig()
    config.d = 0
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

def get_nextlat_config(d=1, **kwargs):
    """Get NextLat config with specified horizon"""
    config = ModelConfig()
    config.d = d
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

def get_mtp_config(d=2, **kwargs):
    """Get MTP baseline config"""
    config = ModelConfig()
    config.mtp_d = d
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config