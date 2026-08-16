"""
Configuration for Tamil Nano/Micro LLM
Optimized for Apple Silicon M4 / 16GB RAM and edge devices.
Architecture: Modern Decoder-Only Transformer (RoPE, SwiGLU, RMSNorm, GQA)
"""
from dataclasses import dataclass, asdict
import json
from typing import Optional


@dataclass
class TamilNanoConfig:
    # Vocabulary & Sequence
    vocab_size: int = 16384           # Tamil-specialized BPE vocabulary size
    max_position_embeddings: int = 1024  # Context window (tokens)
    
    # Model Dimensions (Default: ~60M parameter Micro-LLM)
    hidden_size: int = 512            # Embedding & hidden dimension
    intermediate_size: int = 1376     # SwiGLU FFN hidden dimension (~8/3 * hidden_size)
    num_hidden_layers: int = 10       # Transformer layers
    num_attention_heads: int = 8      # Query heads
    num_key_value_heads: int = 2      # Key/Value heads (Grouped Query Attention - 4:1 ratio)
    
    # Normalization & Activations
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0       # RoPE base frequency
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True  # Tie input embedding & output head to save memory
    dropout: float = 0.0              # Dropout rate (0.0 for LLM pretraining)
    
    # Special Token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TamilNanoConfig":
        return cls(**data)

    def save_pretrained(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, load_path: str) -> "TamilNanoConfig":
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Presets for different size configurations
CONFIG_PRESETS = {
    "nano-25m": TamilNanoConfig(
        vocab_size=8192,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=512,
    ),
    "micro-60m": TamilNanoConfig(
        vocab_size=16384,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=10,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    ),
    "mini-125m": TamilNanoConfig(
        vocab_size=16384,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    ),
}
