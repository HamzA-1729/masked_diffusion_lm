import math
import torch

from masked_diffusion_lm.loss import MDLMLoss
from masked_diffusion_lm.model import MaskedDiffusionLM
from masked_diffusion_lm.noise_schedule import LogLinearSchedule


def test_uniform_predictor_sanity_nelbo():
    """
    A uniform predictor over 4,096 real tokens (mask logit = -inf) has expected
    NELBO = ln(4096) = 8.317766... nats/token (= 12.0 bits/token) because E[(1/t) * t] = 1.
    """
    sched = LogLinearSchedule(t_eps=1e-3)
    criterion = MDLMLoss(sched, mask_token_id=4096, loss_weighting="elbo")
    gen = torch.Generator().manual_seed(42)

    B, L = 128, 256
    x_0 = torch.randint(0, 4096, (B, L), generator=gen)
    t = sched.sample_t(B, generator=gen)
    x_t, _ = sched.q_sample(x_0, t, mask_id=4096, generator=gen)

    logits = torch.zeros(B, L, 4097)
    logits[..., 4096] = float("-inf")

    out = criterion(logits, x_0, x_t, t)
    expected = math.log(4096)
    assert abs(out.nelbo_per_token.item() - expected) < 0.05
    assert abs(out.nelbo_per_token.item() / math.log(2.0) - 12.0) < 0.08


def test_zero_loss_when_no_tokens_masked():
    sched = LogLinearSchedule(t_eps=1e-3)
    criterion = MDLMLoss(sched, mask_token_id=4096)

    x_0 = torch.randint(0, 4096, (4, 16))
    x_t = x_0.clone()  # Zero masked tokens
    t = torch.full((4,), 0.5)
    logits = torch.randn(4, 16, 4097)

    out = criterion(logits, x_0, x_t, t)
    assert out.loss.item() == 0.0
    assert out.masked_fraction == 0.0


def test_gradient_flow_elbo_vs_unweighted():
    sched = LogLinearSchedule(t_eps=1e-3)
    model = MaskedDiffusionLM(depth=1, hidden_dim=32, n_heads=4, n_kv_heads=2, head_dim=8, ffn_dim=64)
    x_0 = torch.randint(0, 4096, (4, 16))
    t = torch.full((4,), 0.5)
    x_t, _ = sched.q_sample(x_0, t, mask_id=4096)

    criterion = MDLMLoss(sched, mask_token_id=4096, loss_weighting="elbo")
    out = criterion(model(x_t), x_0, x_t, t)
    out.loss.backward()

    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0
