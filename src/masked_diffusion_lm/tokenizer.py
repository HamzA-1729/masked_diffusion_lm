"""
Diffusion Tokenizer extending modern_decoder_llm's 4,096-token Byte-Level BPE
with a dedicated absorbing <|mask|> token (id = 4096, vocab_size = 4097).
"""

from collections.abc import Sequence

import torch
from modern_decoder_llm.tokenizer import Tokenizer as BaseTokenizer


class DiffusionTokenizer:
    """
    Wrapper around Project 2's 4,096-vocab Byte-Level BPE tokenizer that reserves
    one dedicated <|mask|> token at ID 4096 without modifying existing token IDs.
    """

    DEFAULT_MASK = "<|mask|>"

    def __init__(self, base_tokenizer: BaseTokenizer, mask_token: str = DEFAULT_MASK):
        self.base = base_tokenizer
        self.mask_token = mask_token

        if mask_token in self.base.special_tokens:
            self.mask_id = self.base.special_tokens[mask_token]
            self.vocab_size = self.base.vocab_size
        else:
            self.mask_id = self.base.vocab_size
            self.vocab_size = self.base.vocab_size + 1

        self.special_tokens: dict[str, int] = dict(self.base.special_tokens)
        self.special_tokens[mask_token] = self.mask_id
        self.inverse_special_tokens: dict[int, str] = {
            v: k for k, v in self.special_tokens.items()
        }

        self.vocab: dict[int, bytes] = dict(self.base.vocab)
        self.vocab[self.mask_id] = mask_token.encode("utf-8")

    @property
    def pad_id(self) -> int:
        return self.base.pad_id

    @property
    def eos_id(self) -> int:
        return self.base.eos_id

    @property
    def bos_id(self) -> int:
        return self.base.bos_id

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Encodes text into token IDs (never emits <|mask|> unless explicitly in text)."""
        return self.base.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decodes token IDs (including <|mask|>) back to a UTF-8 string."""
        byte_chunks: list[bytes] = []
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int == self.mask_id:
                byte_chunks.append(self.mask_token.encode("utf-8"))
            elif tid_int in self.inverse_special_tokens:
                byte_chunks.append(self.inverse_special_tokens[tid_int].encode("utf-8"))
            elif tid_int in self.vocab:
                byte_chunks.append(self.vocab[tid_int])
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def token_byte_lengths(self) -> torch.Tensor:
        """
        Returns a 1D LongTensor of shape (vocab_size,) where entry i is the
        exact UTF-8 byte length of token i (0 for <|mask|>).
        Used as the exact BPB denominator so AR and MDLM byte counts match.
        """
        lengths = torch.zeros(self.vocab_size, dtype=torch.long)
        for tid in range(self.base.vocab_size):
            if tid in self.vocab:
                lengths[tid] = len(self.vocab[tid])
        lengths[self.mask_id] = 0
        return lengths

    def count_bytes(self, token_ids: torch.Tensor) -> int:
        """Computes total UTF-8 bytes for a tensor of token IDs."""
        table = self.token_byte_lengths().to(token_ids.device)
        return int(table[token_ids].sum().item())

    @classmethod
    def load(cls, filepath: str) -> "DiffusionTokenizer":
        """Loads a saved P2 tokenizer.json and appends the <|mask|> token."""
        base = BaseTokenizer.load(filepath)
        return cls(base)
