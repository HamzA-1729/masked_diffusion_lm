"""
Continuous-Time Rao-Blackwellized NELBO Loss for Masked Diffusion (MDLM).

Exact normalization:
    L_NELBO = [ sum_{b,l} 1[x_{t,b,l} == MASK] * (1 / t_b) * CE(p_theta(. | x_{t,b})_l, x_{0,b,l}) ] / (B * L)

Note: Dividing by all (B * L) positions (rather than only masked positions) preserves
the exact continuous-time ELBO expectation E[(1/t) * t] = 1.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from masked_diffusion_lm.noise_schedule import LogLinearSchedule


@dataclass
class LossOutput:
    loss: torch.Tensor
    nelbo_per_token: torch.Tensor
    nelbo_per_seq: torch.Tensor
    masked_fraction: float


class MDLMLoss(nn.Module):
    """Rao-Blackwellized continuous-time MDLM objective with ELBO or unweighted mode."""

    def __init__(
        self,
        schedule: LogLinearSchedule,
        mask_token_id: int = 4096,
        loss_weighting: str = "elbo",
    ):
        super().__init__()
        assert loss_weighting in ("elbo", "unweighted"), (
            f"loss_weighting must be 'elbo' or 'unweighted', got {loss_weighting}"
        )
        self.schedule = schedule
        self.mask_token_id = mask_token_id
        self.loss_weighting = loss_weighting

    def forward(
        self,
        logits: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> LossOutput:
        """
        Args:
            logits: (B, L, V) model logits
            x_0: (B, L) clean target tokens in [0, V - 2]
            x_t: (B, L) noisy input tokens in [0, V - 1]
            t: (B,) noise timesteps in [t_eps, 1.0]
        """
        b_sz, seq_len, vocab_size = logits.shape

        # Enforce SUBS constraint: zero probability of predicting <|mask|>
        safe_logits = logits.clone()
        safe_logits[..., self.mask_token_id] = float("-inf")

        ce = F.cross_entropy(
            safe_logits.view(-1, vocab_size),
            x_0.view(-1),
            reduction="none",
        ).view(b_sz, seq_len)

        is_masked = (x_t == self.mask_token_id).to(ce.dtype)
        weights = self.schedule.elbo_weight(t).view(b_sz, 1).to(ce.dtype)

        # Exact NELBO: sum over masked positions of (1/t) * CE, divided by (B * L)
        weighted_ce = is_masked * weights * ce
        nelbo_per_seq = weighted_ce.sum(dim=1) / float(seq_len)
        nelbo_per_token = weighted_ce.sum() / float(b_sz * seq_len)

        if self.loss_weighting == "elbo":
            loss = nelbo_per_token
        else:
            loss = (is_masked * ce).sum() / float(b_sz * seq_len)

        return LossOutput(
            loss=loss,
            nelbo_per_token=nelbo_per_token.detach(),
            nelbo_per_seq=nelbo_per_seq.detach(),
            masked_fraction=float(is_masked.mean().item()),
        )
