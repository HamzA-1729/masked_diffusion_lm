"""
Evaluation & Benchmarking Suite for Discrete Masked Diffusion LM (MDLM).

Benchmarks:
1. Held-out Likelihood: MDLM Val BPB (<= ELBO) vs. Project 2 exact AR BPB on identical
   non-overlapping val.bin windows (exact byte denominators per scored target).
2. Sample Quality Sweep across [16, 32, 64, 128] steps for Ancestral MDLM and
   Confidence Top-K (extension), reporting Gen PPL (judged by P2 DecoderLLM),
   Unigram Token Entropy (nats, ceiling ln(256) = 5.545), and actual NFE.
3. Reference Baselines: Real held-out TinyStories + P2's own AR samples (T=1.0).
4. Qualitative Arbitrary-Position Infilling Showcase (extension).
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from modern_decoder_llm.model import DecoderLLM
from modern_decoder_llm.sampler import sample_next_token

from masked_diffusion_lm.dataset import DiffusionMemmapDataset, resolve_data_and_tokenizer
from masked_diffusion_lm.loss import MDLMLoss
from masked_diffusion_lm.noise_schedule import LogLinearSchedule
from masked_diffusion_lm.sampler import ancestral_sample, confidence_topk_sample, infill
from masked_diffusion_lm.tokenizer import DiffusionTokenizer
from masked_diffusion_lm.train import build_model_from_config, evaluate_validation_elbo


def compute_unigram_entropy(tokens: torch.Tensor) -> float:
    """
    Per-sample unigram token entropy H(x) = -sum_v p(v) ln p(v) in nats,
    averaged across samples in the batch. Ceiling for L=256 is ln(256) = 5.545.
    """
    entropies = []
    for row in tokens:
        _, counts = torch.unique(row, return_counts=True)
        probs = counts.to(torch.float64) / float(row.numel())
        h = -(probs * torch.log(probs)).sum().item()
        entropies.append(h)
    return float(sum(entropies) / max(len(entropies), 1))


def load_ar_judge(ar_ckpt_path: str, device: torch.device) -> Optional[DecoderLLM]:
    """Loads Project 2's DecoderLLM checkpoint if available."""
    if not os.path.exists(ar_ckpt_path):
        print(f"Warning: AR checkpoint not found at {ar_ckpt_path}; skipping AR Gen PPL / BPB.")
        return None

    ckpt = torch.load(ar_ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).get("model", {})
    ar_model = DecoderLLM(
        vocab_size=cfg.get("vocab_size", 4096),
        max_seq_len=cfg.get("max_seq_len", 256),
        hidden_dim=cfg.get("hidden_dim", 256),
        depth=cfg.get("depth", 6),
        n_heads=cfg.get("n_heads", 8),
        n_kv_heads=cfg.get("n_kv_heads", 2),
        head_dim=cfg.get("head_dim", 32),
        ffn_dim=cfg.get("ffn_dim", 768),
        rope_theta=cfg.get("rope_theta", 10000.0),
        norm_eps=cfg.get("norm_eps", 1e-6),
        tie_weights=cfg.get("tie_weights", True),
    ).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    ar_model.load_state_dict(state_dict)
    ar_model.eval()
    return ar_model


@torch.no_grad()
def evaluate_ar_val_bpb(
    ar_model: DecoderLLM,
    val_loader: DataLoader,
    tokenizer: DiffusionTokenizer,
    max_batches: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates Project 2's exact AR BPB on the 255 shifted targets per 256-token window,
    using the exact UTF-8 byte count of those 255 targets as the denominator.
    """
    byte_table = tokenizer.token_byte_lengths().to(device)
    seq_bpbs = []
    seq_nlls = []

    for b_idx, x_0 in enumerate(val_loader):
        if b_idx >= max_batches:
            break
        x_0 = x_0.to(device)
        inp = x_0[:, :-1]      # (B, 255)
        targets = x_0[:, 1:]   # (B, 255)

        logits, _ = ar_model(inp)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)

        seq_nats = ce.sum(dim=1)
        seq_bytes = byte_table[targets].sum(dim=1).to(torch.float32).clamp(min=1.0)
        seq_bpb = seq_nats / (math.log(2.0) * seq_bytes)

        seq_nlls.append(ce.mean(dim=1).cpu())
        seq_bpbs.append(seq_bpb.cpu())

    all_nll = torch.cat(seq_nlls)
    all_bpb = torch.cat(seq_bpbs)
    n = max(len(all_bpb), 1)
    return {
        "ar_val_nll": float(all_nll.mean().item()),
        "ar_val_bpb": float(all_bpb.mean().item()),
        "ar_val_bpb_se": float(all_bpb.std(unbiased=True).item() / math.sqrt(n)) if n > 1 else 0.0,
    }


@torch.no_grad()
def score_gen_ppl(ar_model: DecoderLLM, tokens: torch.Tensor) -> float:
    """Computes Generative Perplexity exp(mean NLL) under the frozen P2 AR judge."""
    inp = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits, _ = ar_model(inp)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean")
    return float(torch.exp(ce).item())


@torch.no_grad()
def generate_ar_samples(
    ar_model: DecoderLLM,
    batch_size: int,
    seq_len: int,
    eos_id: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """Generates autoregressive samples from P2 DecoderLLM at temperature=1.0, top_p=1.0."""
    torch.manual_seed(seed)
    tokens = torch.full((batch_size, 1), eos_id, dtype=torch.long, device=device)
    for _ in range(seq_len - 1):
        logits, _ = ar_model(tokens)
        next_tok = sample_next_token(logits[:, -1, :], temperature=1.0, top_p=1.0)
        tokens = torch.cat([tokens, next_tok], dim=1)
    return tokens


def plot_sweep(sweep_results: List[Dict], ref_ppl: Optional[float], ref_ent: float, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in ("ancestral", "confidence_topk"):
        rows = [r for r in sweep_results if r["method"] == method]
        steps = [r["steps"] for r in rows]
        if any(r["gen_ppl"] is not None for r in rows):
            ax1.plot(steps, [r["gen_ppl"] for r in rows], marker="o", label=method)
        ax2.plot(steps, [r["entropy"] for r in rows], marker="s", label=method)

    if ref_ppl is not None:
        ax1.axhline(ref_ppl, linestyle="--", color="gray", label="Real Data Ref")
    ax2.axhline(ref_ent, linestyle="--", color="gray", label="Real Data Ref")

    ax1.set_title("Generative Perplexity vs. Steps (lower = more fluent)")
    ax1.set_xlabel("Diffusion Steps")
    ax1.set_ylabel("Gen PPL (P2 AR Judge)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Unigram Token Entropy vs. Steps (ceiling ln 256 = 5.545)")
    ax2.set_xlabel("Diffusion Steps")
    ax2.set_ylabel("Entropy (nats)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_evaluation(cfg: Dict, ckpt_path: str) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_bin, tokenizer = resolve_data_and_tokenizer(cfg)
    seq_len = cfg["model"]["max_seq_len"]
    t_cfg = cfg["training"]
    s_cfg = cfg["sampling"]

    val_ds = DiffusionMemmapDataset(val_bin, seq_len=seq_len)
    val_loader = DataLoader(val_ds, batch_size=t_cfg["batch_size"], shuffle=False, drop_last=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    schedule = LogLinearSchedule(t_eps=cfg["diffusion"]["t_eps"])
    criterion = MDLMLoss(schedule, mask_token_id=tokenizer.mask_id, loss_weighting="elbo")

    # 1. Deterministic Held-Out BPB Comparison
    mdlm_val = evaluate_validation_elbo(
        model, val_loader, schedule, criterion, tokenizer,
        max_batches=t_cfg["eval_batches"], eval_seed=cfg["seed"] + 1000, device=device,
    )
    ar_model = load_ar_judge(cfg["paths"]["ar_checkpoint"], device)
    ar_val = (
        evaluate_ar_val_bpb(ar_model, val_loader, tokenizer, t_cfg["eval_batches"], device)
        if ar_model is not None
        else None
    )

    # 2. Reference Real Data & P2 AR Sample Baselines
    num_samples = s_cfg["num_eval_samples"]
    real_batch = torch.stack([val_ds[i] for i in range(min(num_samples, len(val_ds)))], dim=0).to(device)
    real_entropy = compute_unigram_entropy(real_batch)
    real_gen_ppl = score_gen_ppl(ar_model, real_batch) if ar_model is not None else None

    ar_self_stats = None
    if ar_model is not None:
        ar_samples = generate_ar_samples(ar_model, num_samples, seq_len, tokenizer.eos_id, device)
        ar_self_stats = {
            "gen_ppl": score_gen_ppl(ar_model, ar_samples),
            "entropy": compute_unigram_entropy(ar_samples),
        }

    # 3. Step Sweep: Ancestral MDLM vs. Confidence Top-K (Extension)
    sweep_results = []
    for steps in s_cfg["num_steps_sweep"]:
        for method in ("ancestral", "confidence_topk"):
            gen = torch.Generator(device=device).manual_seed(cfg["seed"] + steps)
            if method == "ancestral":
                res = ancestral_sample(
                    model, schedule, seq_len=seq_len, batch_size=num_samples,
                    num_steps=steps, mask_id=tokenizer.mask_id,
                    temperature=s_cfg["temperature"], use_cache=s_cfg["use_cache"],
                    generator=gen, device=device,
                )
            else:
                res = confidence_topk_sample(
                    model, schedule, seq_len=seq_len, batch_size=num_samples,
                    num_steps=steps, mask_id=tokenizer.mask_id,
                    temperature=s_cfg["temperature"], generator=gen, device=device,
                )

            sweep_results.append({
                "method": method,
                "steps": steps,
                "actual_nfe": res.actual_nfe,
                "gen_ppl": score_gen_ppl(ar_model, res.tokens) if ar_model is not None else None,
                "entropy": compute_unigram_entropy(res.tokens),
            })

    # 4. Infilling Showcase (Extension)
    ref_seq = val_ds[0][:128].clone()
    prompt = ref_seq.clone()
    prompt[32:96] = tokenizer.mask_id
    infill_res = infill(
        model, schedule, prompt.unsqueeze(0).to(device),
        mask_id=tokenizer.mask_id, method="ancestral", num_steps=64,
    )

    report = {
        "mdlm_val": mdlm_val,
        "ar_val": ar_val,
        "real_data_ref": {"gen_ppl": real_gen_ppl, "entropy": real_entropy},
        "ar_self_samples_T1": ar_self_stats,
        "sweep": sweep_results,
        "infill_example": {
            "original": tokenizer.decode(ref_seq.tolist()),
            "infilled": tokenizer.decode(infill_res.tokens[0].cpu().tolist()),
        },
    }

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    plot_sweep(sweep_results, real_gen_ppl, real_entropy, os.path.join(out_dir, "nfe_sweep_plot.png"))
    print(json.dumps({k: v for k, v in report.items() if k != "infill_example"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Masked Diffusion LM")
    parser.add_argument("--config", type=str, default="configs/tinystories_mdlm.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_evaluation(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
