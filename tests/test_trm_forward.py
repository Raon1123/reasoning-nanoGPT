"""
Tests for TRM (Tiny Recursive Model) forward pass.

Covers:
  - Basic forward shape (with and without targets)
  - Weight sharing: single L_level used across all cycles
  - H_cycles=1 (no no_grad path) vs H_cycles>1 (no_grad path)
  - Gradient flows only from the final outer iteration
  - Puzzle embedding integration
  - Loss with ignore_label_id masking
"""
import sys
sys.path.append('.')

import pytest
import torch
import torch.nn as nn

from models.trm import TRM, TRMConfig


def _base_config(**kwargs) -> TRMConfig:
    defaults = dict(
        block_size=16,
        vocab_size=8,
        ignore_label_id=-100,
        H_cycles=2,
        L_cycles=2,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        activation='swiglu',
        expansion=2.667,
        normalize='rmsnorm',
        pos_encodings='rotary',
        rope_theta=10000.0,
        sparsity=1.0,
        puzzle_emb_ndim=0,
        num_identifiers=0,
        batch_size=2,
        forward_dtype=torch.float32,
    )
    defaults.update(kwargs)
    return TRMConfig(**defaults)


def _make_batch(cfg: TRMConfig, b=2, t=8):
    idx = torch.randint(0, cfg.vocab_size, (b, t))
    targets = torch.randint(0, cfg.vocab_size, (b, t))
    return idx, targets


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_forward_no_targets_shape():
    cfg = _base_config()
    model = TRM(cfg)
    model.eval()
    idx, _ = _make_batch(cfg)

    logits, loss = model(idx)

    # Generation mode: only last token
    assert logits.shape == (2, 1, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_shape():
    cfg = _base_config()
    model = TRM(cfg)
    model.eval()
    idx, targets = _make_batch(cfg)

    logits, loss = model(idx, targets=targets)

    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None
    assert loss.ndim == 0  # scalar


def test_forward_h1_l1():
    """H_cycles=1 skips the no_grad path entirely."""
    cfg = _base_config(H_cycles=1, L_cycles=1)
    model = TRM(cfg)
    model.eval()
    idx, targets = _make_batch(cfg)
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None


# ── Weight sharing ─────────────────────────────────────────────────────────────

def test_single_l_level_module():
    """TRM has exactly one L_level (ReasoningModule), not one per cycle."""
    cfg = _base_config(H_cycles=3, L_cycles=4)
    model = TRM(cfg)

    # Only one L_level child
    reasoning_modules = [m for name, m in model.named_children() if name == 'L_level']
    assert len(reasoning_modules) == 1


def test_param_count_independent_of_cycles():
    """Parameter count must NOT grow with more H/L cycles (weight sharing)."""
    cfg_a = _base_config(H_cycles=1, L_cycles=1)
    cfg_b = _base_config(H_cycles=5, L_cycles=10)

    params_a = sum(p.numel() for p in TRM(cfg_a).parameters())
    params_b = sum(p.numel() for p in TRM(cfg_b).parameters())

    assert params_a == params_b, (
        f"Param count should be identical regardless of cycles: {params_a} != {params_b}"
    )


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_gradient_flows_to_l_level():
    cfg = _base_config(H_cycles=2, L_cycles=2)
    model = TRM(cfg)
    model.train()
    idx, targets = _make_batch(cfg)

    _, loss = model(idx, targets=targets)
    loss.backward()

    for name, p in model.L_level.named_parameters():
        assert p.grad is not None, f"No grad for L_level.{name}"
        assert not torch.all(p.grad == 0), f"Zero grad for L_level.{name}"


def test_gradient_flows_to_wte():
    cfg = _base_config(H_cycles=2, L_cycles=2)
    model = TRM(cfg)
    model.train()
    idx, targets = _make_batch(cfg)

    _, loss = model(idx, targets=targets)
    loss.backward()

    # wte and lm_head share weights
    assert model.wte.weight.grad is not None


# ── No-grad path ───────────────────────────────────────────────────────────────

def test_nograd_path_executes_for_h_cycles_gt_1():
    """When H_cycles>1, the first H_cycles-1 iters run under no_grad.
    Verify output is still valid (no error) and gradients still exist after backward."""
    cfg = _base_config(H_cycles=3, L_cycles=2)
    model = TRM(cfg)
    model.train()
    idx, targets = _make_batch(cfg)

    logits, loss = model(idx, targets=targets)
    loss.backward()

    # Gradients should still exist (from the final grad iteration)
    grads = [p.grad for p in model.L_level.parameters()]
    assert all(g is not None for g in grads)


# ── Loss masking ───────────────────────────────────────────────────────────────

def test_loss_with_all_ignored_labels():
    """When all targets are IGNORE, PyTorch cross_entropy returns NaN (no valid tokens).
    Verify the forward still runs without crashing and returns a scalar."""
    cfg = _base_config()
    model = TRM(cfg)
    model.eval()
    idx, _ = _make_batch(cfg)
    targets = torch.full((2, 8), cfg.ignore_label_id, dtype=torch.long)

    logits, loss = model(idx, targets=targets)
    # PyTorch cross_entropy returns NaN when all labels are ignored — this is expected.
    assert loss is not None
    assert loss.ndim == 0  # must still be a scalar tensor


def test_loss_with_partial_ignore():
    """Partial ignoring should give a valid scalar loss."""
    cfg = _base_config()
    model = TRM(cfg)
    model.eval()
    idx, targets = _make_batch(cfg)
    targets[:, :4] = cfg.ignore_label_id  # mask first half

    _, loss = model(idx, targets=targets)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


# ── Rotary vs learned positional encodings ─────────────────────────────────────

def test_learned_pos_enc():
    cfg = _base_config(pos_encodings='learned')
    model = TRM(cfg)
    model.eval()
    idx, targets = _make_batch(cfg)
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_rotary_pos_enc():
    cfg = _base_config(pos_encodings='rotary')
    model = TRM(cfg)
    model.eval()
    idx, targets = _make_batch(cfg)
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 8, cfg.vocab_size)


# ── Puzzle embedding ───────────────────────────────────────────────────────────

def test_puzzle_emb_forward():
    """With puzzle_emb_ndim>0, forward should still produce correct-shaped logits."""
    cfg = _base_config(puzzle_emb_ndim=16, num_identifiers=10, batch_size=2)
    model = TRM(cfg)
    model.eval()

    b, t = 2, 8
    idx = torch.randint(0, cfg.vocab_size, (b, t))
    targets = torch.randint(0, cfg.vocab_size, (b, t))
    puzzle_idx = torch.randint(0, cfg.num_identifiers, (b,))

    logits, loss = model(idx, puzzle_idx=puzzle_idx, targets=targets)
    # logits should be stripped of puzzle prefix — shape (b, t, vocab_size)
    assert logits.shape == (b, t, cfg.vocab_size)
    assert loss is not None


# ── Determinism ────────────────────────────────────────────────────────────────

def test_forward_deterministic_in_eval():
    """Same input in eval mode should produce identical output."""
    cfg = _base_config()
    model = TRM(cfg)
    model.eval()
    idx, targets = _make_batch(cfg)

    with torch.no_grad():
        logits1, loss1 = model(idx, targets=targets)
        logits2, loss2 = model(idx, targets=targets)

    assert torch.allclose(logits1, logits2)
    assert torch.allclose(loss1, loss2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
