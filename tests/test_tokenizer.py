import os
import tempfile

from modern_decoder_llm.tokenizer import Tokenizer as BaseTokenizer

from masked_diffusion_lm.tokenizer import DiffusionTokenizer


def _make_trained_base_tokenizer() -> BaseTokenizer:
    tok = BaseTokenizer(vocab_size=300)
    tok.train(
        [
            "Once upon a time, there was a little bird named Timmy.",
            "Timmy loved to fly across the green forest and sing songs.",
        ],
        min_freq=1,
    )
    return tok


def test_mask_token_reserved_without_collision():
    base = _make_trained_base_tokenizer()
    dtok = DiffusionTokenizer(base)

    assert dtok.vocab_size == base.vocab_size + 1
    assert dtok.mask_id == base.vocab_size
    assert dtok.mask_id != dtok.eos_id
    assert dtok.mask_id != dtok.pad_id


def test_roundtrip_and_byte_length_parity():
    base = _make_trained_base_tokenizer()
    dtok = DiffusionTokenizer(base)

    sample = "Once upon a time, Timmy flew!<|endoftext|>"
    ids = dtok.encode(sample)
    assert dtok.decode(ids) == sample

    byte_table = dtok.token_byte_lengths()
    summed_bytes = sum(int(byte_table[tid].item()) for tid in ids)
    assert summed_bytes == len(sample.encode("utf-8"))
    assert int(byte_table[dtok.mask_id].item()) == 0


def test_load_from_json_file():
    base = _make_trained_base_tokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokenizer.json")
        base.save(path)
        dtok = DiffusionTokenizer.load(path)
        assert dtok.vocab_size == base.vocab_size + 1
        assert dtok.decode([dtok.mask_id]) == "<|mask|>"
