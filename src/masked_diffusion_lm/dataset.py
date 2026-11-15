"""
Memory-Mapped Dataset Pipeline for Discrete Masked Diffusion Pretraining.

Unlike autoregressive models (which shift windows by 1 token to form (x, y) pairs),
MDLM trains on unshifted, fixed-length token windows x_0 of shape (seq_len,).
Reads the same uint16 binary shards as modern_decoder_llm for exact data parity.
"""

import os
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from modern_decoder_llm.dataset import prepare_tinystories_data
from masked_diffusion_lm.tokenizer import DiffusionTokenizer


class DiffusionMemmapDataset(Dataset):
    """
    PyTorch Dataset backed by a memory-mapped uint16 binary token file.
    Yields non-overlapping windows of length `seq_len` as int64 tensors.
    """

    def __init__(self, bin_path: str, seq_len: int = 256):
        super().__init__()
        assert os.path.exists(bin_path), f"Binary dataset file not found: {bin_path}"
        self.bin_path = bin_path
        self.seq_len = seq_len

        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        assert self.total_tokens >= self.seq_len, (
            f"Dataset has {self.total_tokens} tokens, but requires at least {self.seq_len}"
        )
        self.num_samples = self.total_tokens // self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.data[start:end].astype(np.int64)
        return torch.from_numpy(chunk)


def resolve_data_and_tokenizer(cfg: Dict) -> Tuple[str, str, DiffusionTokenizer]:
    """
    Resolves TinyStories binary shards and the 4,096-BPE tokenizer (+ <|mask|>).
    Prefers Project 2's existing artifacts when available; otherwise prepares local shards.
    """
    paths = cfg["paths"]
    tok_path = paths["tokenizer_path"]
    train_bin = paths["train_bin"]
    val_bin = paths["val_bin"]

    if os.path.exists(tok_path) and os.path.exists(train_bin) and os.path.exists(val_bin):
        tokenizer = DiffusionTokenizer.load(tok_path)
        return train_bin, val_bin, tokenizer

    fallback_dir = paths.get("fallback_data_dir", "data")
    fallback_tok = paths.get("fallback_tokenizer_path", "checkpoints/tokenizer.json")
    local_train, local_val, base_tok = prepare_tinystories_data(
        output_dir=fallback_dir,
        tokenizer_path=fallback_tok,
        vocab_size=cfg["model"]["vocab_size"] - 1,
    )
    return local_train, local_val, DiffusionTokenizer(base_tok)
