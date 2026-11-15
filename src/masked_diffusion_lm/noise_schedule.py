"""
Continuous-Time Log-Linear Noise Schedule for Masked Diffusion (MDLM Appendix E.1).

Definition:
    sigma(t) = -log(1 - t)
    alpha(t) = exp(-sigma(t)) = 1 - t     (survival probability of remaining unmasked)
    alpha'(t) = -1
    ELBO weight: -alpha'(t) / (1 - alpha(t)) = 1 / t

Includes the low-discrepancy stratified timestep sampler from MDLM Appendix D.3:
    u_i ~ Uniform[(i - 1) / B, i / B],   t_i = eps + (1 - eps) * u_i
"""


import torch


class LogLinearSchedule:
    """Log-linear absorbing-state diffusion schedule with clamped t >= t_eps."""

    def __init__(self, t_eps: float = 1e-3):
        assert 0.0 < t_eps < 1.0, f"t_eps must be in (0, 1), got {t_eps}"
        self.t_eps = float(t_eps)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Survival probability alpha_t = 1 - t (probability a token stays unmasked)."""
        return 1.0 - t.clamp(min=0.0, max=1.0)

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Masking probability 1 - alpha_t = t."""
        return t.clamp(min=self.t_eps, max=1.0)

    def elbo_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Continuous-time NELBO weight: -alpha'(t) / (1 - alpha(t)) = 1 / t."""
        return 1.0 / t.clamp(min=self.t_eps, max=1.0)

    def sample_t(
        self,
        batch_size: int,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Draws batch_size stratified timesteps in [t_eps, 1.0] (MDLM Appendix D.3).
        Each sample i in 0..B-1 is drawn uniformly from [i/B, (i+1)/B].
        """
        u = torch.rand(batch_size, device=device, generator=generator, dtype=torch.float32)
        bins = torch.arange(batch_size, device=device, dtype=torch.float32)
        stratified = (bins + u) / float(batch_size)
        return self.t_eps + (1.0 - self.t_eps) * stratified

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask_id: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward absorbing process q(x_t | x_0):
        Independently replaces each token with `mask_id` with probability 1 - alpha_t = t.

        Returns:
            x_t: Noisy token tensor of shape (B, L)
            is_masked: Boolean mask of shape (B, L) where True indicates masked tokens
        """
        p_mask = self.mask_prob(t).view(-1, 1).to(x_0.device)
        rand = torch.rand(x_0.shape, device=x_0.device, generator=generator, dtype=torch.float32)
        is_masked = rand < p_mask
        x_t = torch.where(is_masked, torch.full_like(x_0, mask_id), x_0)
        return x_t, is_masked
