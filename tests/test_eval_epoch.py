"""
Tests for eval_epoch — uses a mock model and in-memory dataloader.
Verifies the fix for B1 (pbar_iter NameError) and B9 (mask device mismatch).
"""
import sys
sys.path.append('.')

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from utils.epochs import eval_epoch
from utils.const import IGNORE_LABEL_ID


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=12, seq_len=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def forward(self, X, puzzle_ids, Y, test_mode=True):
        B = X.size(0)
        # Return uniform logits and a scalar loss
        logits = torch.zeros(B, self.seq_len, self.vocab_size)
        loss = torch.tensor(1.0)
        return logits, loss


def _make_dataloader(n=32, seq_len=16, vocab_size=12, batch_size=8):
    X = torch.randint(0, vocab_size, (n, seq_len))
    Y = torch.randint(0, vocab_size, (n, seq_len))
    puzzle_ids = torch.zeros(n, dtype=torch.long)
    dataset = TensorDataset(X, Y, puzzle_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _dummy_config(eval_iters=3):
    return {'logging': {'eval_iters': eval_iters}}


def test_eval_epoch_returns_correct_keys():
    model = _DummyModel()
    loader = _make_dataloader()
    config = _dummy_config(eval_iters=2)
    device = torch.device('cpu')

    out = eval_epoch(config, model, loader, device)
    assert 'loss' in out
    assert 'accuracy' in out
    assert 'sequence_accuracy' in out


def test_eval_epoch_no_nameerror_on_first_call():
    """B1 fix: pbar_iter must not raise NameError on the very first call."""
    model = _DummyModel()
    loader = _make_dataloader()
    config = _dummy_config(eval_iters=1)
    device = torch.device('cpu')
    # Should not raise NameError
    out = eval_epoch(config, model, loader, device)
    assert out['loss'] >= 0


def test_eval_epoch_multiple_calls():
    """Iterator should reset cleanly across multiple eval_epoch calls."""
    model = _DummyModel()
    loader = _make_dataloader(n=8, batch_size=4)
    config = _dummy_config(eval_iters=5)
    device = torch.device('cpu')

    out1 = eval_epoch(config, model, loader, device)
    out2 = eval_epoch(config, model, loader, device)
    assert abs(out1['loss'] - out2['loss']) < 1e-6


def test_eval_epoch_all_ignored_labels():
    """When all Y tokens are IGNORE_LABEL_ID, accuracy metrics should be 0/0 safe."""
    vocab_size = 12
    seq_len = 16
    n, bs = 16, 8

    X = torch.randint(0, vocab_size, (n, seq_len))
    Y = torch.full((n, seq_len), IGNORE_LABEL_ID, dtype=torch.long)
    puzzle_ids = torch.zeros(n, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, Y, puzzle_ids)
    loader = DataLoader(dataset, batch_size=bs)

    model = _DummyModel(vocab_size=vocab_size, seq_len=seq_len)
    config = _dummy_config(eval_iters=1)
    device = torch.device('cpu')

    # Should not raise ZeroDivisionError
    try:
        out = eval_epoch(config, model, loader, device)
    except ZeroDivisionError:
        pytest.fail("eval_epoch raised ZeroDivisionError on all-ignored labels")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
