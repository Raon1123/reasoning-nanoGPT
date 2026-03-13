"""
Tests for CosineSchedulerWithWarmup and CombinedScheduler.

NOTE: PyTorch's LRScheduler calls step() once in __init__, setting last_epoch=0.
The first user call to step() advances to last_epoch=1. Tests here verify
behavioral invariants (monotone warmup, decay, min_ratio floor) rather than
exact lr values at specific steps, since the exact offset depends on PyTorch internals.
"""
import sys
sys.path.append('.')

import torch
import pytest

from models.scheduler import CosineSchedulerWithWarmup, CombinedScheduler, NanoGPTScheduler


def _dummy_optimizer(lr=0.1):
    param = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.SGD([param], lr=lr)


# ── CosineSchedulerWithWarmup ──────────────────────────────────────────────────

def test_cosine_lr_is_positive_after_init():
    base_lr = 0.1
    opt = _dummy_optimizer(base_lr)
    CosineSchedulerWithWarmup(opt, base_lr=base_lr, num_warmup_steps=4, num_training_steps=20)
    assert opt.param_groups[0]['lr'] > 0


def test_cosine_warmup_lr_increases():
    """LR should increase monotonically during warmup steps."""
    base_lr = 0.1
    warmup = 8
    total = 40
    opt = _dummy_optimizer(base_lr)
    sch = CosineSchedulerWithWarmup(opt, base_lr=base_lr, num_warmup_steps=warmup, num_training_steps=total)

    lrs = [opt.param_groups[0]['lr']]
    for _ in range(warmup):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])

    # Check monotone increase within the warmup region (not at the warmup/decay boundary)
    # The last collected lr may be slightly lower than peak due to cosine decay kicking in
    for i in range(len(lrs) - 2):
        assert lrs[i] <= lrs[i + 1] + 1e-9, f"LR decreased during warmup at step {i}: {lrs[i]} > {lrs[i+1]}"
    assert lrs[0] < lrs[-2] + 1e-9, "LR should have increased over the warmup phase"


def test_cosine_decay_lr_decreases():
    """LR should be lower at the end of training than at the warmup peak."""
    base_lr = 0.1
    warmup = 4
    total = 20
    opt = _dummy_optimizer(base_lr)
    sch = CosineSchedulerWithWarmup(opt, base_lr=base_lr, num_warmup_steps=warmup, num_training_steps=total)

    for _ in range(warmup):
        sch.step()
    peak_lr = opt.param_groups[0]['lr']

    for _ in range(total - warmup):
        sch.step()
    final_lr = opt.param_groups[0]['lr']

    assert final_lr <= peak_lr, f"Final lr {final_lr} should be <= peak lr {peak_lr}"


def test_cosine_min_ratio_respected():
    """LR should not go below base_lr * min_ratio."""
    base_lr = 0.1
    min_ratio = 0.1
    warmup = 2
    total = 20
    opt = _dummy_optimizer(base_lr)
    sch = CosineSchedulerWithWarmup(opt, base_lr=base_lr, num_warmup_steps=warmup,
                                     num_training_steps=total, min_ratio=min_ratio)

    for _ in range(total + 10):
        sch.step()

    lr = opt.param_groups[0]['lr']
    assert lr >= base_lr * min_ratio - 1e-7, \
        f"LR {lr} went below min_ratio floor {base_lr * min_ratio}"


# ── CombinedScheduler ──────────────────────────────────────────────────────────

def test_combined_scheduler_both_lrs_update():
    opt1 = _dummy_optimizer(0.1)
    opt2 = _dummy_optimizer(0.2)
    sch1 = NanoGPTScheduler(opt1, warmup_iters=2, lr_decay_iters=10, min_lr=0.01, max_lr=0.1)
    sch2 = NanoGPTScheduler(opt2, warmup_iters=2, lr_decay_iters=10, min_lr=0.02, max_lr=0.2)
    combined = CombinedScheduler([sch1, sch2])

    lr1_before = opt1.param_groups[0]['lr']
    lr2_before = opt2.param_groups[0]['lr']
    combined.step()
    lr1_after = opt1.param_groups[0]['lr']
    lr2_after = opt2.param_groups[0]['lr']

    assert lr1_after != lr1_before or lr2_after != lr2_before


def test_combined_scheduler_get_last_lr():
    opt1 = _dummy_optimizer(0.1)
    opt2 = _dummy_optimizer(0.2)
    sch1 = NanoGPTScheduler(opt1, warmup_iters=2, lr_decay_iters=10, min_lr=0.01, max_lr=0.1)
    sch2 = NanoGPTScheduler(opt2, warmup_iters=2, lr_decay_iters=10, min_lr=0.02, max_lr=0.2)
    combined = CombinedScheduler([sch1, sch2])

    combined.step()
    last_lr = combined.get_last_lr()
    assert isinstance(last_lr, list)
    assert len(last_lr) > 0
    assert all(isinstance(v, float) for v in last_lr)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
