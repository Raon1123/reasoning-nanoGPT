"""
Tests for CombinedOptimizer and CastedSparseEmbeddingSignSGD_Distributed.
"""
import sys
sys.path.append('.')

import torch
import pytest

from models.optimizer import CombinedOptimizer, CastedSparseEmbeddingSignSGD_Distributed


def _make_simple_model():
    return torch.nn.Linear(8, 4)


def test_combined_optimizer_step():
    m1 = _make_simple_model()
    m2 = _make_simple_model()
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
    combined = CombinedOptimizer(opt1, opt2)

    x = torch.randn(2, 8)
    loss = m1(x).sum() + m2(x).sum()
    loss.backward()
    combined.step()
    combined.zero_grad()

    for p in list(m1.parameters()) + list(m2.parameters()):
        assert p.grad is None


def test_combined_optimizer_state_dict_roundtrip():
    m1 = _make_simple_model()
    m2 = _make_simple_model()
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    combined = CombinedOptimizer(opt1, opt2)

    # Take a step to populate state
    x = torch.randn(2, 8)
    (m1(x).sum() + m2(x).sum()).backward()
    combined.step()

    sd = combined.state_dict()
    assert 'opt1' in sd and 'opt2' in sd

    # Load into fresh combined optimizer
    m1b = _make_simple_model()
    m2b = _make_simple_model()
    opt1b = torch.optim.Adam(m1b.parameters(), lr=1e-3)
    opt2b = torch.optim.Adam(m2b.parameters(), lr=1e-3)
    combined2 = CombinedOptimizer(opt1b, opt2b)
    combined2.load_state_dict(sd)

    sd2 = combined2.state_dict()
    assert sd['opt1'].keys() == sd2['opt1'].keys()


def test_combined_optimizer_get_last_lr():
    """CombinedOptimizer.get_last_lr() delegates to opt2.get_last_lr(),
    which is provided by the scheduler wrapping opt2."""
    from models.scheduler import NanoGPTScheduler, CombinedScheduler
    m1 = _make_simple_model()
    m2 = _make_simple_model()
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.02)
    combined = CombinedOptimizer(opt1, opt2)

    # Attach schedulers — get_last_lr is provided by the scheduler, not the optimizer
    sch1 = NanoGPTScheduler(opt1, warmup_iters=2, lr_decay_iters=10, min_lr=0.001, max_lr=0.01)
    sch2 = NanoGPTScheduler(opt2, warmup_iters=2, lr_decay_iters=10, min_lr=0.002, max_lr=0.02)
    combined_sch = CombinedScheduler([sch1, sch2])

    x = torch.randn(2, 8)
    (m1(x).sum() + m2(x).sum()).backward()
    combined.step()
    combined_sch.step()

    last_lr = combined_sch.get_last_lr()
    assert isinstance(last_lr, list)
    assert len(last_lr) > 0
    assert all(v > 0 for v in last_lr)


def test_signsgd_single_process():
    """CastedSparseEmbeddingSignSGD_Distributed with world_size=1."""
    num_emb, emb_dim, batch = 20, 8, 4
    from models.layers import CastedSparseEmbedding
    module = CastedSparseEmbedding(
        num_embeddings=num_emb,
        embedding_dim=emb_dim,
        batch_size=batch,
        init_std=0.01,
        cast_to=torch.float32,
    )
    module.train()

    optimizer = CastedSparseEmbeddingSignSGD_Distributed(
        module.parameters(),
        lr=1e-2,
        weight_decay=0.01,
        world_size=1,
    )

    ids = torch.randint(0, num_emb, (batch,))
    emb = module(ids)
    loss = emb.sum()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
