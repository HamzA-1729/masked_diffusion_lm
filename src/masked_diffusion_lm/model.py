"""
Bidirectional Llama-3-Style Transformer for Discrete Masked Diffusion (MDLM).

Reuses RMSNorm, RoPE (rotate_half), and SwiGLU from modern_decoder_llm for strict
architectural parity with Project 2, replacing causal attention with full
bidirectional Grouped-Query Attention (no KV cache).

When time_conditioning=False (default), parameter count is 5,574,144
(Project 2's 5,573,888 + 256 for the single <|mask|> embedding row).
"""

import math

import torch
import torch.nn.functional as F
from modern_decoder_llm.model import (
    RMSNorm,
    SwiGLU,
    apply_rotary_emb,
    precompute_rope_freqs,
)
from torch import nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for continuous time t in [0, 1], used only if time_conditioning=True."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class BidirectionalGQA(nn.Module):
    """
    Bidirectional Grouped-Query Attention (8 Q-heads, 2 KV-heads, 4:1 ratio).
    Identical projections and RoPE layout to modern_decoder_llm, without causal masking.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 32,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        assert n_heads * head_dim == hidden_dim

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b_sz, seq_len, _ = x.shape

        q = self.q_proj(x).view(b_sz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b_sz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b_sz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin, start_pos=0)
        k = apply_rotary_emb(k, cos, sin, start_pos=0)

        if self.num_groups > 1:
            k = torch.repeat_interleave(k, repeats=self.num_groups, dim=1)
            v = torch.repeat_interleave(v, repeats=self.num_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(b_sz, seq_len, self.hidden_dim)
        return self.o_proj(output)


class BidirectionalBlock(nn.Module):
    """Pre-RMSNorm Bidirectional Transformer block with optional AdaLN-Zero time conditioning."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 32,
        ffn_dim: int = 768,
        norm_eps: float = 1e-6,
        time_conditioning: bool = False,
    ):
        super().__init__()
        self.time_conditioning = time_conditioning
        self.attn_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.attn = BidirectionalGQA(hidden_dim, n_heads, n_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn = SwiGLU(hidden_dim, ffn_dim)

        if time_conditioning:
            self.ada_ln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
            )
            nn.init.zeros_(self.ada_ln[1].weight)
            nn.init.zeros_(self.ada_ln[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        t_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.time_conditioning:
            assert t_emb is not None, "t_emb required when time_conditioning=True"
            mod = self.ada_ln(t_emb).unsqueeze(1)  # (B, 1, 6 * D)
            g1, b1, a1, g2, b2, a2 = mod.chunk(6, dim=-1)
            h_attn = self.attn_norm(x) * (1.0 + g1) + b1
            x = x + a1 * self.attn(h_attn, cos, sin)
            h_ffn = self.ffn_norm(x) * (1.0 + g2) + b2
            x = x + a2 * self.ffn(h_ffn)
            return x

        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MaskedDiffusionLM(nn.Module):
    """
    Discrete Masked Diffusion Language Model with SUBS parameterization.
    """

    def __init__(
        self,
        vocab_size: int = 4097,
        mask_token_id: int = 4096,
        max_seq_len: int = 256,
        hidden_dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 32,
        ffn_dim: int = 768,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        tie_weights: bool = True,
        time_conditioning: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.tie_weights = tie_weights
        self.time_conditioning = time_conditioning

        self.tok_embeddings = nn.Embedding(vocab_size, hidden_dim)

        cos, sin = precompute_rope_freqs(head_dim, max_seq_len, theta_base=rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        if time_conditioning:
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.layers = nn.ModuleList([
            BidirectionalBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                ffn_dim=ffn_dim,
                norm_eps=norm_eps,
                time_conditioning=time_conditioning,
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self) -> None:
        scaled_std = 0.02 / math.sqrt(2.0 * self.depth)
        for name, p in self.named_parameters():
            if "ada_ln" in name:
                continue  # Keep AdaLN-Zero identity initialization
            if "weight" in name and p.dim() >= 2:
                if "o_proj" in name or "down_proj" in name:
                    nn.init.normal_(p, mean=0.0, std=scaled_std)
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        t: torch.Tensor | None = None,
        apply_subs: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) LongTensor with values in [0, vocab_size - 1]
            t: Optional (B,) noise level tensor (required only if time_conditioning=True)
            apply_subs: If True, zeroes out probability of predicting <|mask|> (logit = -inf)

        Returns:
            logits: (B, L, vocab_size)
        """
        _, seq_len = token_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len ({self.max_seq_len})"
        )

        h = self.tok_embeddings(token_ids)
        t_emb = None
        if self.time_conditioning:
            assert t is not None, "Timestep t must be provided when time_conditioning=True"
            t_emb = self.time_mlp(sinusoidal_time_embedding(t, self.hidden_dim))

        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin, t_emb=t_emb)

        h = self.norm(h)
        logits = self.lm_head(h)

        if apply_subs:
            logits = logits.clone()
            logits[..., self.mask_token_id] = float("-inf")

        return logits


def subs_log_probs(logits: torch.Tensor, x_t: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """
    Full SUBS parameterization log-probabilities:
    1. <|mask|> logit is -inf (zero probability of predicting [MASK]).
    2. Unmasked positions (x_t != mask_token_id) carry through x_t with probability 1.0.
    """
    masked_logits = logits.clone()
    masked_logits[..., mask_token_id] = float("-inf")
    log_probs = F.log_softmax(masked_logits, dim=-1)

    unmasked = x_t != mask_token_id
    if unmasked.any():
        carry_over = torch.full_like(log_probs, float("-inf"))
        carry_over.scatter_(-1, x_t.unsqueeze(-1), 0.0)
        log_probs = torch.where(unmasked.unsqueeze(-1), carry_over, log_probs)

    return log_probs
