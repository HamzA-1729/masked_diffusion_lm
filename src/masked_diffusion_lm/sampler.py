"""
Reverse Samplers for Discrete Masked Diffusion Language Models (MDLM).

Includes:
1. Exact MDLM Ancestral Sampler with prediction caching (Sahoo et al., 2024).
2. Confidence Top-K Sampler (MaskGIT-style heuristic extension; not exact posterior).
3. Arbitrary-Position Infilling (extension clamping known prefix/suffix/middle tokens).

All categorical / Gumbel draws are executed in torch.float64 to prevent the float32
Gumbel truncation bug (Zheng et al., 2024) that artificially collapses token diversity.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

from masked_diffusion_lm.model import MaskedDiffusionLM
from masked_diffusion_lm.noise_schedule import LogLinearSchedule


@dataclass
class SampleResult:
    tokens: torch.Tensor
    actual_nfe: int
    num_steps: int


def draw_uniform_float64(
    shape: tuple[int, ...],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draws uniform random numbers in [0, 1) strictly in torch.float64."""
    return torch.rand(shape, device=device, dtype=torch.float64, generator=generator)


def sample_categorical_float64(
    probs: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Samples token IDs from categorical probabilities using the Gumbel-max trick
    strictly in torch.float64 (avoids float32 Gumbel truncation bias).
    """
    probs_f64 = probs.to(torch.float64).clamp(min=1e-30)
    u = draw_uniform_float64(probs_f64.shape, device=probs_f64.device, generator=generator)
    gumbel = -torch.log(-torch.log(u.clamp(min=1e-30)))
    return torch.argmax(torch.log(probs_f64) + gumbel, dim=-1)


@torch.no_grad()
def ancestral_sample(
    model: MaskedDiffusionLM,
    schedule: LogLinearSchedule,
    seq_len: int = 256,
    batch_size: int = 1,
    num_steps: int = 64,
    mask_id: int = 4096,
    temperature: float = 1.0,
    use_cache: bool = True,
    initial_tokens: Optional[torch.Tensor] = None,
    clamp_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> SampleResult:
    """
    Exact MDLM ancestral reverse sampler.

    At step t -> s:
      - Unmasked tokens stay fixed (SUBS carry-over).
      - Masked tokens unmask with probability (alpha_s - alpha_t) / (1 - alpha_t) = (t - s) / t,
        sampling from p_theta(x_0^i | x_t) in float64.
      - Final step (t -> 0) samples all remaining [MASK] tokens from p_theta in float64 (no argmax).
      - When time_conditioning=False and no token unmasked on the previous step, prediction caching
        reuses p_theta without a forward pass while still advancing RNG identically.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if initial_tokens is not None:
        x = initial_tokens.clone().to(device)
        b_sz, l_len = x.shape
    else:
        b_sz, l_len = batch_size, seq_len
        x = torch.full((b_sz, l_len), mask_id, dtype=torch.long, device=device)

    if clamp_mask is None:
        clamped = torch.zeros((b_sz, l_len), dtype=torch.bool, device=device)
    else:
        clamped = clamp_mask.to(device=device, dtype=torch.bool)

    ts = torch.linspace(1.0, schedule.t_eps, num_steps + 1, device=device)
    cached_probs: Optional[torch.Tensor] = None
    changed_prev_step = True
    actual_nfe = 0

    for step_idx in range(num_steps):
        t_val = float(ts[step_idx].item())
        s_val = float(ts[step_idx + 1].item())

        can_reuse = (
            use_cache
            and (not model.time_conditioning)
            and (cached_probs is not None)
            and (not changed_prev_step)
        )
        if can_reuse:
            probs = cached_probs
        else:
            t_tensor = torch.full((b_sz,), t_val, device=device, dtype=torch.float32)
            logits = model(x, t=t_tensor if model.time_conditioning else None, apply_subs=True)
            actual_nfe += 1
            scaled_logits = (logits / max(temperature, 1e-6)).to(torch.float64)
            probs = F.softmax(scaled_logits, dim=-1)
            cached_probs = probs

        # Always draw RNG every step so use_cache=True and use_cache=False are bit-identical
        u_unmask = draw_uniform_float64((b_sz, l_len), device=device, generator=generator)
        sampled_tokens = sample_categorical_float64(probs, generator=generator)

        is_mask = (x == mask_id) & (~clamped)
        if step_idx == num_steps - 1:
            unmask_now = is_mask
        else:
            p_unmask = (t_val - s_val) / max(t_val, schedule.t_eps)
            unmask_now = is_mask & (u_unmask < p_unmask)

        x = torch.where(unmask_now, sampled_tokens, x)
        changed_prev_step = bool(unmask_now.any().item())

    return SampleResult(tokens=x, actual_nfe=actual_nfe, num_steps=num_steps)


@torch.no_grad()
def confidence_topk_sample(
    model: MaskedDiffusionLM,
    schedule: LogLinearSchedule,
    seq_len: int = 256,
    batch_size: int = 1,
    num_steps: int = 64,
    mask_id: int = 4096,
    temperature: float = 1.0,
    gumbel_noise_scale: float = 0.0,
    initial_tokens: Optional[torch.Tensor] = None,
    clamp_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> SampleResult:
    """
    Extension (MaskGIT-style heuristic, NOT exact reverse posterior):
    Unmasks top-k highest-confidence masked tokens per step matching the linear
    ancestral unmask count floor(L_unclamped * (1 - s)).
    Candidate tokens are sampled in float64; confidence is p_theta(cand | x_t).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if initial_tokens is not None:
        x = initial_tokens.clone().to(device)
        b_sz, l_len = x.shape
    else:
        b_sz, l_len = batch_size, seq_len
        x = torch.full((b_sz, l_len), mask_id, dtype=torch.long, device=device)

    if clamp_mask is None:
        clamped = torch.zeros((b_sz, l_len), dtype=torch.bool, device=device)
    else:
        clamped = clamp_mask.to(device=device, dtype=torch.bool)

    ts = torch.linspace(1.0, schedule.t_eps, num_steps + 1, device=device)
    unclamped_counts = (~clamped).sum(dim=1)  # (B,)
    actual_nfe = 0

    for step_idx in range(num_steps):
        t_val = float(ts[step_idx].item())
        s_val = float(ts[step_idx + 1].item())

        t_tensor = torch.full((b_sz,), t_val, device=device, dtype=torch.float32)
        logits = model(x, t=t_tensor if model.time_conditioning else None, apply_subs=True)
        actual_nfe += 1

        probs = F.softmax((logits / max(temperature, 1e-6)).to(torch.float64), dim=-1)
        cand_tokens = sample_categorical_float64(probs, generator=generator)
        conf = probs.gather(-1, cand_tokens.unsqueeze(-1)).squeeze(-1)  # float64

        if gumbel_noise_scale > 0.0:
            u = draw_uniform_float64(conf.shape, device=device, generator=generator)
            gumbel = -torch.log(-torch.log(u.clamp(min=1e-30)))
            anneal = 1.0 - (step_idx / float(num_steps))
            conf = torch.log(conf.clamp(min=1e-30)) + gumbel_noise_scale * anneal * gumbel

        is_mask = (x == mask_id) & (~clamped)
        if step_idx == num_steps - 1:
            x = torch.where(is_mask, cand_tokens, x)
            continue

        # Mask out already-unmasked or clamped positions from top-k selection
        masked_conf = torch.where(is_mask, conf, torch.full_like(conf, float("-inf")))
        for b in range(b_sz):
            n_avail = int(is_mask[b].sum().item())
            if n_avail == 0:
                continue
            already_unmasked = int(unclamped_counts[b].item()) - n_avail
            target_unmasked = int(round(float(unclamped_counts[b].item()) * (1.0 - s_val)))
            k = min(n_avail, max(0, target_unmasked - already_unmasked))
            if k > 0:
                topk_idx = torch.topk(masked_conf[b], k=k).indices
                x[b, topk_idx] = cand_tokens[b, topk_idx]

    return SampleResult(tokens=x, actual_nfe=actual_nfe, num_steps=num_steps)


@torch.no_grad()
def infill(
    model: MaskedDiffusionLM,
    schedule: LogLinearSchedule,
    prompt_tokens: torch.Tensor,
    mask_id: int = 4096,
    method: str = "ancestral",
    num_steps: int = 64,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> SampleResult:
    """
    Extension: Arbitrary-position infilling.
    Clamps all positions where prompt_tokens != mask_id and denoises masked spans.
    """
    clamp_mask = prompt_tokens != mask_id
    if method == "ancestral":
        return ancestral_sample(
            model=model,
            schedule=schedule,
            num_steps=num_steps,
            mask_id=mask_id,
            temperature=temperature,
            initial_tokens=prompt_tokens,
            clamp_mask=clamp_mask,
            generator=generator,
        )
    if method == "confidence_topk":
        return confidence_topk_sample(
            model=model,
            schedule=schedule,
            num_steps=num_steps,
            mask_id=mask_id,
            temperature=temperature,
            initial_tokens=prompt_tokens,
            clamp_mask=clamp_mask,
            generator=generator,
        )
    raise ValueError(f"Unknown infilling method: {method}")
