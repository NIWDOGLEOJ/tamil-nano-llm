"""
PyTorch Implementation of Tamil Nano/Micro LLM
Optimized for Apple Silicon Metal (MPS) and CUDA.
Architecture: LLaMA/Qwen-style Decoder (RMSNorm, RoPE, SwiGLU, Grouped-Query Attention)
"""
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import TamilNanoConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # High precision calculation for variance
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        return normed.type_as(x) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes Rotary Positional Embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Return polar representation in complex space
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # [max_seq_len, head_dim // 2]


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary position embeddings to query or key tensors.
    x shape: [batch, seq_len, num_heads, head_dim]
    """
    # Reshape x to complex view: pairs of (even, odd) coordinates
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis: [seq_len, head_dim // 2] -> broadcast to [1, seq_len, 1, head_dim // 2]
    freqs_cis = freqs_cis[: x.shape[1], :].unsqueeze(0).unsqueeze(2)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.type_as(x)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: TamilNanoConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TamilNanoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_key_value_groups

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Update / Handle KV cache for autoregressive inference
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
        new_kv_cache = (k, v) if self.training is False else None

        # Repeat KV heads for Grouped Query Attention (GQA)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # Transpose to [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlashAttention / Scaled Dot-Product Attention (MPS & CUDA accelerated)
        causal = is_causal and seq_len > 1 and (kv_cache is None or kv_cache[0].shape[1] == 0)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        # Transpose back: [batch, seq_len, heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output), new_kv_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: TamilNanoConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Attention with residual
        normed_x = self.input_layernorm(x)
        attn_out, new_cache = self.self_attn(normed_x, freqs_cis, kv_cache=kv_cache)
        x = x + attn_out

        # Pre-LN MLP with residual
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


class TamilNanoForCausalLM(nn.Module):
    def __init__(self, config: TamilNanoConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying (shares weights between embedding & lm_head to halve parameter count)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies
        freqs_cis = precompute_rope_freqs(config.head_dim, config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Note: if tied, embed_tokens is shared
        return {
            "total": total,
            "trainable": trainable,
            "total_millions": round(total / 1e6, 2),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        return_dict: bool = True,
    ):
        batch_size, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis.to(x.device)

        new_caches = [] if kv_caches is not None else None

        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            x, new_cache = layer(x, freqs_cis, kv_cache=layer_cache)
            if new_caches is not None:
                new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return {
            "loss": loss,
            "logits": logits,
            "kv_caches": new_caches,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with sampling controls."""
        self.eval()
        eos_id = eos_token_id or self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            cond_input = input_ids if input_ids.shape[1] <= self.config.max_position_embeddings else input_ids[:, -self.config.max_position_embeddings:]
            
            outputs = self(cond_input)
            next_token_logits = outputs["logits"][:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    score = next_token_logits[0, token_id]
                    next_token_logits[0, token_id] = score / repetition_penalty if score > 0 else score * repetition_penalty

            if temperature <= 0.0:
                # Greedy search
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Top-K / Top-P sampling
                logits = next_token_logits / max(temperature, 1e-5)
                
                # Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            token_val = next_token.item()
            if token_val == eos_id or (hasattr(self.config, 'im_end_id') and token_val == self.config.im_end_id):
                break

        return input_ids

