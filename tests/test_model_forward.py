"""
Tests for GPT model forward pass.
Covers: shapes, loss computation, sparsity masking, separate_reasoning mode.
All tests use synthetic (random) data — no disk fixtures needed.
"""
import sys
sys.path.append('.')

import torch
import pytest

from models.nanogpt import GPT, GPTConfig
from utils.const import IGNORE_LABEL_ID


def make_config(**overrides):
    defaults = dict(
        block_size=16,
        vocab_size=12,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
        sparsity=1.0,
        puzzle_emb_ndim=0,
        num_identifiers=0,
        ignore_label_id=IGNORE_LABEL_ID,
        batch_size=4,
        activation='gelu',
        normalize='layernorm',
        pos_encodings='learned',
        separate_reasoning=False,
        forward_dtype=torch.float32,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def test_forward_no_targets():
    cfg = make_config()
    model = GPT(cfg)
    model.eval()
    B, T = 4, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, 1, cfg.vocab_size), f"Expected (B,1,V), got {logits.shape}"
    assert loss is None


def test_forward_with_targets_shapes():
    cfg = make_config()
    model = GPT(cfg)
    model.eval()
    B, T = 4, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is not None
    assert loss.ndim == 0  # scalar


def test_forward_loss_ignores_padding():
    cfg = make_config()
    model = GPT(cfg)
    model.eval()
    B, T = 4, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.full((B, T), IGNORE_LABEL_ID, dtype=torch.long)
    # All targets are ignored — cross_entropy on empty mask should not raise
    # (PyTorch returns NaN or 0 depending on version; just check no error)
    with torch.no_grad():
        logits, loss = model(idx, targets=targets)
    assert loss is not None


def test_forward_with_puzzle_embeddings():
    B = 4
    cfg = make_config(
        puzzle_emb_ndim=32,
        num_identifiers=10,
        batch_size=B,
    )
    model = GPT(cfg)
    model.eval()
    T = 12
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    puzzle_idx = torch.randint(0, cfg.num_identifiers, (B,))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, puzzle_idx=puzzle_idx, targets=targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is not None


def test_forward_separate_reasoning():
    cfg = make_config(separate_reasoning=True)
    model = GPT(cfg)
    model.eval()
    B, T = 4, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is not None


def test_forward_rotary_pos_encoding():
    cfg = make_config(pos_encodings='rotary')
    model = GPT(cfg)
    model.eval()
    B, T = 4, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, cfg.vocab_size)


def test_forward_sparsity():
    cfg = make_config(sparsity=0.5)
    model = GPT(cfg)
    model.train()
    B, T = 4, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets, test_mode=False)
    assert loss is not None
    # In test_mode, sparsity is disabled
    logits2, loss2 = model(idx, targets=targets, test_mode=True)
    assert loss2 is not None


def test_forward_rmsnorm():
    cfg = make_config(normalize='rmsnorm')
    model = GPT(cfg)
    model.eval()
    B, T = 4, 10
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, cfg.vocab_size)


def test_get_num_params():
    cfg = make_config()
    model = GPT(cfg)
    n = model.get_num_params()
    assert n > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
