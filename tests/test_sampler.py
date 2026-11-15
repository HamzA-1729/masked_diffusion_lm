import torch

from masked_diffusion_lm.model import MaskedDiffusionLM
from masked_diffusion_lm.noise_schedule import LogLinearSchedule
from masked_diffusion_lm.sampler import (
    ancestral_sample,
    confidence_topk_sample,
    draw_uniform_float64,
    infill,
)


def _tiny_model() -> MaskedDiffusionLM:
    torch.manual_seed(0)
    return MaskedDiffusionLM(
        vocab_size=4097,
        mask_token_id=4096,
        max_seq_len=64,
        hidden_dim=32,
        depth=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        ffn_dim=64,
        time_conditioning=False,
    )


def test_uniform_draws_are_float64():
    u = draw_uniform_float64((4, 16), device=torch.device("cpu"))
    assert u.dtype == torch.float64


def test_ancestral_cache_equivalence_and_zero_masks():
    model = _tiny_model()
    sched = LogLinearSchedule(t_eps=1e-3)

    gen_cached = torch.Generator().manual_seed(77)
    res_cached = ancestral_sample(
        model, sched, seq_len=16, batch_size=2, num_steps=32, use_cache=True, generator=gen_cached
    )

    gen_uncached = torch.Generator().manual_seed(77)
    res_uncached = ancestral_sample(
        model, sched, seq_len=16, batch_size=2, num_steps=32, use_cache=False, generator=gen_uncached
    )

    assert torch.equal(res_cached.tokens, res_uncached.tokens)
    assert (res_cached.tokens != 4096).all()
    assert (res_cached.tokens >= 0).all() and (res_cached.tokens < 4096).all()


def test_prediction_cache_reduces_nfe_at_high_steps():
    """At T=1024 on L=16, many steps unmask zero tokens, so actual_nfe < num_steps."""
    model = _tiny_model()
    sched = LogLinearSchedule(t_eps=1e-3)
    gen = torch.Generator().manual_seed(11)

    res = ancestral_sample(
        model, sched, seq_len=16, batch_size=1, num_steps=1024, use_cache=True, generator=gen
    )
    assert res.actual_nfe < res.num_steps
    assert (res.tokens != 4096).all()


def test_confidence_topk_produces_valid_tokens():
    model = _tiny_model()
    sched = LogLinearSchedule(t_eps=1e-3)
    gen = torch.Generator().manual_seed(5)

    res = confidence_topk_sample(model, sched, seq_len=16, batch_size=2, num_steps=16, generator=gen)
    assert res.tokens.shape == (2, 16)
    assert (res.tokens != 4096).all()


def test_infill_preserves_clamped_prefix_and_suffix():
    model = _tiny_model()
    sched = LogLinearSchedule(t_eps=1e-3)

    prompt = torch.full((1, 16), 4096, dtype=torch.long)
    prefix = torch.tensor([10, 20, 30, 40])
    suffix = torch.tensor([100, 200, 300, 400])
    prompt[0, :4] = prefix
    prompt[0, 12:] = suffix

    res = infill(model, sched, prompt, method="ancestral", num_steps=16)
    assert torch.equal(res.tokens[0, :4], prefix)
    assert torch.equal(res.tokens[0, 12:], suffix)
    assert (res.tokens != 4096).all()
