import torch

from masked_diffusion_lm.noise_schedule import LogLinearSchedule


def test_schedule_bounds_and_weight():
    sched = LogLinearSchedule(t_eps=1e-3)

    t_zero = torch.tensor([0.0])
    t_one = torch.tensor([1.0])
    assert torch.isclose(sched.alpha(t_zero), torch.tensor([1.0]))
    assert torch.isclose(sched.alpha(t_one), torch.tensor([0.0]))

    # Clamping prevents 1/t blowup at t < t_eps
    assert torch.isclose(sched.elbo_weight(t_zero), torch.tensor([1000.0]))
    assert torch.isclose(sched.elbo_weight(torch.tensor([0.5])), torch.tensor([2.0]))


def test_stratified_sampler_covers_bins():
    sched = LogLinearSchedule(t_eps=1e-3)
    gen = torch.Generator().manual_seed(42)
    B = 16
    t = sched.sample_t(B, generator=gen)

    assert t.shape == (B,)
    assert (t >= 1e-3).all() and (t <= 1.0).all()

    # Each sample i must lie in [t_eps + (1 - t_eps)*(i/B), t_eps + (1 - t_eps)*((i+1)/B)]
    for i in range(B):
        lo = 1e-3 + (1.0 - 1e-3) * (i / B)
        hi = 1e-3 + (1.0 - 1e-3) * ((i + 1) / B)
        assert lo <= float(t[i]) <= hi


def test_q_sample_empirical_mask_rate():
    sched = LogLinearSchedule(t_eps=1e-3)
    gen = torch.Generator().manual_seed(123)
    x_0 = torch.randint(0, 4096, (32, 256))
    t = torch.full((32,), 0.25)

    x_t, is_masked = sched.q_sample(x_0, t, mask_id=4096, generator=gen)
    empirical_rate = is_masked.float().mean().item()
    assert abs(empirical_rate - 0.25) < 0.03
    assert (x_t[is_masked] == 4096).all()
    assert (x_t[~is_masked] == x_0[~is_masked]).all()
