"""
Training Pipeline for Discrete Masked Diffusion Language Model (MDLM).

Features:
- Selective AdamW weight decay (0.1 on 2D linear matrices only; 0.0 on RMSNorm/embeddings).
- Linear warmup + cosine LR decay matching Project 2 (modern_decoder_llm).
- Deterministic validation on non-overlapping val.bin windows with a fixed-seed t grid,
  reporting Val NELBO ± SE and Val BPB (<= ELBO) ± SE.
"""

import argparse
import json
import math
import os

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from masked_diffusion_lm.dataset import DiffusionMemmapDataset, resolve_data_and_tokenizer
from masked_diffusion_lm.loss import MDLMLoss
from masked_diffusion_lm.model import MaskedDiffusionLM
from masked_diffusion_lm.noise_schedule import LogLinearSchedule
from masked_diffusion_lm.tokenizer import DiffusionTokenizer


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Applies weight decay only to 2D+ linear weights (excludes norms & embeddings)."""
    decay_params = []
    no_decay_params = []
    seen_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        if param.dim() >= 2 and "tok_embeddings" not in name and "lm_head" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(warmup_steps, 1))
    if step >= max_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate_validation_elbo(
    model: MaskedDiffusionLM,
    val_loader: DataLoader,
    schedule: LogLinearSchedule,
    criterion: MDLMLoss,
    tokenizer: DiffusionTokenizer,
    max_batches: int,
    eval_seed: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Deterministic validation: evaluates non-overlapping val windows with a fixed-seed
    stratified t grid and returns Val NELBO ± SE and Val BPB (<= ELBO) ± SE.
    """
    model.eval()
    byte_table = tokenizer.token_byte_lengths().to(device)
    gen = torch.Generator(device="cpu").manual_seed(eval_seed)

    seq_nelbos = []
    seq_bpbs = []

    for b_idx, x_0 in enumerate(val_loader):
        if b_idx >= max_batches:
            break
        x_0 = x_0.to(device)
        b_sz, seq_len = x_0.shape

        t_cpu = schedule.sample_t(b_sz, generator=gen)
        x_t_cpu, _ = schedule.q_sample(x_0.cpu(), t_cpu, mask_id=tokenizer.mask_id, generator=gen)

        t = t_cpu.to(device)
        x_t = x_t_cpu.to(device)

        logits = model(x_t, t=t if model.time_conditioning else None, apply_subs=True)
        out = criterion(logits, x_0, x_t, t)

        # Per-sequence nats = nelbo_per_seq * seq_len; exact UTF-8 bytes per sequence
        seq_nats = out.nelbo_per_seq * float(seq_len)
        seq_bytes = byte_table[x_0].sum(dim=1).to(torch.float32).clamp(min=1.0)
        seq_bpb = seq_nats / (math.log(2.0) * seq_bytes)

        seq_nelbos.append(out.nelbo_per_seq.cpu())
        seq_bpbs.append(seq_bpb.cpu())

    all_nelbo = torch.cat(seq_nelbos)
    all_bpb = torch.cat(seq_bpbs)
    n = max(len(all_nelbo), 1)

    return {
        "val_nelbo": float(all_nelbo.mean().item()),
        "val_nelbo_se": float(all_nelbo.std(unbiased=True).item() / math.sqrt(n)) if n > 1 else 0.0,
        "val_bpb": float(all_bpb.mean().item()),
        "val_bpb_se": float(all_bpb.std(unbiased=True).item() / math.sqrt(n)) if n > 1 else 0.0,
        "num_sequences": n,
    }


def build_model_from_config(cfg: dict) -> MaskedDiffusionLM:
    m = cfg["model"]
    return MaskedDiffusionLM(
        vocab_size=m["vocab_size"],
        mask_token_id=m["mask_token_id"],
        max_seq_len=m["max_seq_len"],
        hidden_dim=m["hidden_dim"],
        depth=m["depth"],
        n_heads=m["n_heads"],
        n_kv_heads=m["n_kv_heads"],
        head_dim=m["head_dim"],
        ffn_dim=m["ffn_dim"],
        rope_theta=m["rope_theta"],
        norm_eps=m["norm_eps"],
        tie_weights=m["tie_weights"],
        time_conditioning=m.get("time_conditioning", False),
    )


def train(cfg: dict, max_steps_override: int | None = None) -> tuple[MaskedDiffusionLM, list[dict]]:
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_bin, val_bin, tokenizer = resolve_data_and_tokenizer(cfg)
    seq_len = cfg["model"]["max_seq_len"]
    t_cfg = cfg["training"]
    d_cfg = cfg["diffusion"]

    train_ds = DiffusionMemmapDataset(train_bin, seq_len=seq_len)
    val_ds = DiffusionMemmapDataset(val_bin, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=t_cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    model = build_model_from_config(cfg).to(device)
    schedule = LogLinearSchedule(t_eps=d_cfg["t_eps"])
    criterion = MDLMLoss(
        schedule=schedule,
        mask_token_id=tokenizer.mask_id,
        loss_weighting=d_cfg.get("loss_weighting", "elbo"),
    )
    optimizer = build_optimizer(model, lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])

    max_steps = max_steps_override if max_steps_override is not None else t_cfg["max_steps"]
    grad_accum = t_cfg["grad_accum_steps"]
    eval_interval = t_cfg["eval_interval"]
    eval_batches = t_cfg["eval_batches"]

    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    history: list[dict] = []
    best_val_bpb = float("inf")
    train_iter = iter(train_loader)

    pbar = tqdm(range(max_steps), desc="Training MDLM")
    for step in pbar:
        lr = get_cosine_lr(step, t_cfg["warmup_steps"], max_steps, t_cfg["lr"], t_cfg["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_nelbo = 0.0

        for _ in range(grad_accum):
            try:
                x_0 = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_0 = next(train_iter)

            x_0 = x_0.to(device)
            t = schedule.sample_t(x_0.shape[0], device=device)
            x_t, _ = schedule.q_sample(x_0, t, mask_id=tokenizer.mask_id)

            logits = model(x_t, t=t if model.time_conditioning else None, apply_subs=True)
            out = criterion(logits, x_0, x_t, t)
            (out.loss / float(grad_accum)).backward()

            accum_loss += float(out.loss.item()) / float(grad_accum)
            accum_nelbo += float(out.nelbo_per_token.item()) / float(grad_accum)

        nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"])
        optimizer.step()

        pbar.set_postfix(loss=f"{accum_loss:.4f}", nelbo=f"{accum_nelbo:.4f}", lr=f"{lr:.2e}")

        if (step + 1) % eval_interval == 0 or step == max_steps - 1:
            val_stats = evaluate_validation_elbo(
                model=model,
                val_loader=val_loader,
                schedule=schedule,
                criterion=criterion,
                tokenizer=tokenizer,
                max_batches=eval_batches,
                eval_seed=seed + 1000,
                device=device,
            )
            record = {
                "step": step + 1,
                "lr": lr,
                "train_loss": accum_loss,
                "train_nelbo": accum_nelbo,
                **val_stats,
            }
            history.append(record)
            print(
                f"\n[Step {step + 1}] Train NELBO: {accum_nelbo:.4f} | "
                f"Val NELBO: {val_stats['val_nelbo']:.4f} ± {val_stats['val_nelbo_se']:.4f} | "
                f"Val BPB (<= ELBO): {val_stats['val_bpb']:.4f} ± {val_stats['val_bpb_se']:.4f}"
            )

            if val_stats["val_bpb"] < best_val_bpb:
                best_val_bpb = val_stats["val_bpb"]
                torch.save(
                    {
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "val_nelbo": val_stats["val_nelbo"],
                        "val_bpb": val_stats["val_bpb"],
                    },
                    os.path.join(ckpt_dir, "best_model.pt"),
                )

    with open(os.path.join(out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Discrete Masked Diffusion LM (MDLM)")
    parser.add_argument("--config", type=str, default="configs/tinystories_mdlm.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
