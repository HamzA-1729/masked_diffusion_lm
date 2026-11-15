import torch

from masked_diffusion_lm.model import MaskedDiffusionLM, subs_log_probs


def test_parameter_count_parity_with_p2():
    model = MaskedDiffusionLM(
        vocab_size=4097,
        mask_token_id=4096,
        max_seq_len=256,
        hidden_dim=256,
        depth=6,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=768,
        tie_weights=True,
        time_conditioning=False,
    )
    total_params = sum(p.numel() for p in model.parameters())
    # P2 has 5,573,888 params; MDLM adds 1 embedding row (256) for <|mask|>
    assert total_params == 5_574_144


def test_time_conditioning_adds_parameters():
    model_t = MaskedDiffusionLM(time_conditioning=True)
    total_params = sum(p.numel() for p in model_t.parameters())
    assert total_params > 5_574_144


def test_bidirectional_non_causality():
    torch.manual_seed(0)
    model = MaskedDiffusionLM(depth=2, hidden_dim=64, n_heads=4, n_kv_heads=2, head_dim=16, ffn_dim=128)
    model.eval()

    x_a = torch.randint(0, 4096, (1, 16))
    x_b = x_a.clone()
    x_b[0, -1] = (x_b[0, -1] + 1) % 4096  # Perturb only the last token

    with torch.no_grad():
        logits_a = model(x_a)
        logits_b = model(x_b)

    # Position 0 MUST depend on position -1 in a bidirectional model
    # Slice [:4096] to exclude the -inf <|mask|> logit ((-inf) - (-inf) = nan)
    diff = (logits_a[0, 0, :4096] - logits_b[0, 0, :4096]).abs().max().item()
    assert diff > 1e-4


def test_subs_zeroes_mask_prob_and_copies_unmasked():
    model = MaskedDiffusionLM(depth=1, hidden_dim=64, n_heads=4, n_kv_heads=2, head_dim=16, ffn_dim=128)
    x_t = torch.tensor([[10, 4096, 25, 4096]])
    logits = model(x_t, apply_subs=True)

    assert logits.shape == (1, 4, 4097)
    assert torch.isinf(logits[..., 4096]).all() and (logits[..., 4096] < 0).all()

    log_probs = subs_log_probs(logits, x_t, mask_token_id=4096)
    probs = log_probs.exp()
    assert torch.isclose(probs[0, 0, 10], torch.tensor(1.0))
    assert torch.isclose(probs[0, 2, 25], torch.tensor(1.0))
    assert torch.isclose(probs[0, 1, 4096], torch.tensor(0.0))
