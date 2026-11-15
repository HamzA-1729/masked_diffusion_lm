import torch

from masked_diffusion_lm.loss import MDLMLoss
from masked_diffusion_lm.model import MaskedDiffusionLM
from masked_diffusion_lm.noise_schedule import LogLinearSchedule


def test_single_batch_overfit_convergence():
    """
    Verifies >95% masked-token reconstruction accuracy at fixed t = 0.5
    and substantial NELBO reduction on a single deterministic batch.
    """
    torch.manual_seed(42)
    sched = LogLinearSchedule(t_eps=1e-3)
    model = MaskedDiffusionLM(
        vocab_size=4097,
        mask_token_id=4096,
        max_seq_len=32,
        hidden_dim=64,
        depth=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_dim=128,
        tie_weights=True,
        time_conditioning=False,
    )
    criterion = MDLMLoss(sched, mask_token_id=4096, loss_weighting="elbo")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    x_0 = torch.randint(0, 4096, (4, 32))
    t_fixed = torch.full((4,), 0.5)
    gen = torch.Generator().manual_seed(99)
    x_t, is_masked = sched.q_sample(x_0, t_fixed, mask_id=4096, generator=gen)

    # Ensure at least some tokens are masked
    assert is_masked.sum().item() > 0

    model.train()
    initial_loss = None
    final_loss = None
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_t, apply_subs=True)
        out = criterion(logits, x_0, x_t, t_fixed)
        if initial_loss is None:
            initial_loss = out.loss.item()
        final_loss = out.loss.item()
        out.loss.backward()
        optimizer.step()

    assert final_loss < initial_loss * 0.05

    model.eval()
    with torch.no_grad():
        preds = model(x_t, apply_subs=True).argmax(dim=-1)
        masked_acc = (preds[is_masked] == x_0[is_masked]).float().mean().item()
    assert masked_acc > 0.95
