# Discrete Masked Diffusion Language Model (MDLM) from Scratch

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, first-principles PyTorch implementation of a **Continuous-Time Discrete Masked Diffusion Language Model (MDLM)** (Sahoo et al., 2024) built from scratch. Designed for an apples-to-apples comparison against the autoregressive (AR) `DecoderLLM` in [`modern_decoder_llm`](https://github.com/HamzA-1729/modern_decoder_llm) (Project 2), sharing the exact TinyStories corpus, 4,096-token Byte-Level BPE tokenizer (`+ <|mask|>`), matched 6-layer Llama-3-style backbone (`5,574,144` parameters), and matched training compute.

---

## 🌟 Core Capabilities & Key Highlights

* **Matched Bidirectional Backbone (`5,574,144` params):** Reuses `RMSNorm`, `RoPE` (`rotate_half`), `SwiGLU` ($d_{\text{ffn}}=768$), and 4:1 `GroupedQueryAttention` directly from `modern_decoder_llm` with full bidirectional attention (no causal mask, no KV cache), differing from Project 2's `5,573,888` parameters by exactly one 256-dim embedding row for `<|mask|>`.
* **SUBS Parameterization & Continuous-Time NELBO:** Sets the `<|mask|>` output logit to $-\infty$, carries unmasked tokens through unchanged, and trains with the exact Rao-Blackwellized continuous-time ELBO using a log-linear schedule ($\alpha_t = 1 - t$) and MDLM Appendix D.3 stratified $t$ sampling.
* **`float64` Categorical Sampling:** Performs all Gumbel-max categorical draws in `torch.float64` to avoid the `float32` Gumbel truncation bug (Zheng et al., 2024) that artificially collapses token diversity and inflates Generative Perplexity.
* **Apples-to-Apples BPB & Sample-Quality Evaluation:** Compares MDLM's held-out **Bits Per Byte ($\le \text{NELBO}$)** against Project 2's exact **AR BPB** using exact scored-token byte denominators on identical non-overlapping `val.bin` windows, alongside a `[16, 32, 64, 128]`-step sweep of **Gen PPL** (scored by Project 2's `DecoderLLM`) and **Unigram Token Entropy**.
* **Extensions Beyond MDLM:** Includes a **Confidence Top-K** heuristic sampler (MaskGIT-style, no likelihood claims) and **Arbitrary-Position Infilling** (clamping fixed prefix/suffix tokens).

---

## 📐 Architecture: Bidirectional Transformer + SUBS

```text
Clean Sequence x_0: (B, L) in [0, 4095]
       │
       ├──► Stratified t_i ~ Uniform[(i-1)/B, i/B] mapped to [eps, 1.0]
       │         └──► Masking Prob: 1 - alpha_t = t
       │
       ▼  Forward Absorbing Process q(x_t | x_0)
Noisy Sequence x_t: (B, L) in [0, 4096]  (4096 = <|mask|>)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Bidirectional Transformer (6 layers, 256d, 8Q/2KV GQA) │
│   • RMSNorm + RoPE (rotate_half) + SwiGLU (768d)       │
│   • Full Bidirectional Attention (no causal mask)      │
│   • Tied Embeddings & LM Head (4097 x 256)             │
└────────────────────────────────────────────────────────┘
       │
       ▼  SUBS Parameterization
  • logits[..., <|mask|>] = -inf
  • Unmasked positions: carry through x_t unchanged
  • Masked positions:   cross-entropy against x_0
       │
       ▼
Rao-Blackwellized NELBO = sum_masked[(1/t) * CE] / (B * L)
```

---

## ⚖️ Autoregressive (`modern_decoder_llm`) vs. Masked Diffusion (`masked_diffusion_lm`)

| Dimension | Autoregressive (`DecoderLLM`) | Masked Diffusion (`MaskedDiffusionLM`) |
| :--- | :--- | :--- |
| **Attention Mask** | Causal lower-triangular | Full bidirectional (all-to-all) |
| **Inference Cache** | Static KV Cache ($O(T)$ per token) | No KV Cache (full-sequence forward per step; prediction cache only skips when 0 tokens unmask) |
| **Likelihood Metric** | Exact NLL / BPB (255 shifted targets) | Upper bound $\le \text{NELBO}$ / BPB (all 256 tokens) |
| **Generation Order** | Strict left-to-right | Parallel iterative unmasking (supports arbitrary infilling) |
| **Compute Efficiency** | Every token receives gradient every step | Only masked positions receive gradient; MDLM BPB trails AR BPB at matched compute |

> [!NOTE]
> **On Likelihood & Prediction Caching:** At matched compute (`2,500` steps, `batch_size=16 x 2`), MDLM's BPB ($\le \text{NELBO}$) is expected to trail Project 2's exact AR BPB because only masked positions contribute gradient signal per step and the ELBO is an upper bound on NLL. Additionally, when sampling with $T \le L$ steps ($T \in \{16, 32, 64, 128\}, L = 256$), prediction caching saves almost no forward passes (~0% at $T \le 32$, ~14% at $T = 128$) because at least one token almost always unmasks per step.

---

## 📚 Mathematical Formulations

### 1. Log-Linear Absorbing Process (MDLM Appendix E.1)
With noise schedule $\sigma(t) = -\log(1 - t)$, the survival probability of a token remaining unmasked at time $t \in [\varepsilon, 1]$ is $\alpha_t = e^{-\sigma(t)} = 1 - t$ ($\alpha_t' = -1$). Each token is independently replaced with `<|mask|>` with probability $1 - \alpha_t = t$.

### 2. Rao-Blackwellized Continuous-Time NELBO
Under SUBS parameterization (`logits[..., <|mask|>] = -inf`, unmasked tokens carried through), the reconstruction and prior terms vanish, leaving:
$$\mathcal{L}_{\mathrm{NELBO}}(\theta) = \frac{1}{B \cdot L} \sum_{b=1}^{B} \sum_{l=1}^{L} \mathbf{1}\big[x_{t,b,l} = \text{<|mask|>}\big] \cdot \frac{1}{t_b} \cdot \mathrm{CE}\big(p_\theta(\cdot \mid x_{t,b})_l, \, x_{0,b,l}\big)$$
For a uniform predictor over the $4{,}096$ real tokens, $\mathbb{E}[(1/t) \cdot t] = 1$, yielding $\mathcal{L}_{\mathrm{NELBO}} = \ln(4096) \approx 8.318\text{ nats/token}$ ($12.0\text{ bits/token}$).

### 3. Bits Per Byte (BPB) Parity
$$\mathrm{BPB} = \frac{\sum \mathrm{NLL}_{\text{nats}}}{\ln(2) \cdot \sum \text{UTF-8 bytes}}$$
Calculated over identical non-overlapping `val.bin` windows using the exact byte length of the scored tokens ($255$ shifted targets for AR; all $256$ tokens for MDLM).

### 4. Exact Ancestral Reverse Sampler (`float64`)
For step $t \to s$ ($s < t$), unmasked tokens remain fixed; each `<|mask|>` token unmasks with probability $\frac{\alpha_s - \alpha_t}{1 - \alpha_t} = \frac{t - s}{t}$ by sampling from $p_\theta(x_0^i \mid x_t)$ in `torch.float64`. At the final step ($t \to 0$), all remaining `<|mask|>` tokens are sampled from $p_\theta$ in `torch.float64` (never `argmax`).

### 5. Extensions (Beyond the MDLM Paper)
* **Confidence Top-K Sampler (`confidence_topk_sample`):** MaskGIT-style heuristic that samples candidate tokens in `float64` and unmasks the top-$k$ highest-confidence positions per step matching the linear schedule. Not an exact reverse posterior; no likelihood claims are made for it.
* **Arbitrary-Position Infilling (`infill`):** Clamps known prefix/suffix/middle tokens (`clamp_mask`) and runs either sampler on the remaining `<|mask|>` spans.

---

## 📁 Repository Structure

```text
masked_diffusion_lm/
├── README.md
├── pyproject.toml                 # uv config depending on modern-decoder-llm @ v0.1.0
├── LICENSE
├── CONTRIBUTING.md
├── CITATION.cff
├── .pre-commit-config.yaml
├── configs/
│   └── tinystories_mdlm.yaml      # Matched P2 compute & backbone configuration
├── checkpoints/                   # Saved MDLM weights (git-ignored)
├── outputs/                       # Eval reports & NFE sweep plots (git-ignored)
├── src/masked_diffusion_lm/
│   ├── __init__.py
│   ├── tokenizer.py               # 4,096 BPE + <|mask|> (id=4096) & byte-length table
│   ├── dataset.py                 # Unshifted uint16 memmap dataset
│   ├── noise_schedule.py          # Log-linear schedule & Appendix D.3 stratified t sampler
│   ├── model.py                   # Bidirectional GQA + RMSNorm + RoPE + SwiGLU + SUBS
│   ├── loss.py                    # Continuous-time (1/t)*CE / (B*L) NELBO loss
│   ├── sampler.py                 # Ancestral (float64 + cache), Confidence Top-K, Infilling
│   ├── train.py                   # Cosine LR + selective weight decay + deterministic val
│   └── evaluate.py                # BPB vs P2 AR, Gen PPL/Entropy sweep, Infilling
└── tests/
    ├── test_tokenizer.py
    ├── test_noise_schedule.py
    ├── test_model.py
    ├── test_loss.py
    ├── test_sampler.py
    └── test_overfit.py
```

---

## 🚀 Quickstart & Reproducibility

```bash
git clone https://github.com/HamzA-1729/masked_diffusion_lm.git
cd masked_diffusion_lm
uv sync
uv run pytest -v
```

### Training & Evaluation
```bash
# Train MDLM on TinyStories (2,500 steps, matched P2 compute)
uv run python -m masked_diffusion_lm.train --config configs/tinystories_mdlm.yaml

# Evaluate BPB (<= ELBO) vs P2 AR BPB, Gen PPL & Entropy sweep, and Infilling
uv run python -m masked_diffusion_lm.evaluate --config configs/tinystories_mdlm.yaml --checkpoint checkpoints/best_model.pt
```

---

## 📖 References

1. **Simple and Effective Masked Diffusion Language Models (MDLM)** — Sahoo et al., 2024 ([arXiv:2406.07524](https://arxiv.org/abs/2406.07524))
2. **Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling** — Zheng et al., 2024 ([arXiv:2409.02908](https://arxiv.org/abs/2409.02908))
3. **Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)** — Austin et al., 2021 ([arXiv:2107.03006](https://arxiv.org/abs/2107.03006))
4. **MaskGIT: Masked Generative Image Transformer** — Chang et al., 2022 ([arXiv:2202.04200](https://arxiv.org/abs/2202.04200))

---

## 📜 License
Distributed under the [MIT License](LICENSE).
