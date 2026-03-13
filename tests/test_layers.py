"""
Tests for individual layers: LayerNorm, CausalSelfAttention, MLP, Block,
SwiGLU, CastedEmbedding, CastedSparseEmbedding, RotaryEmbedding.
"""
import sys
sys.path.append('.')

import torch
import pytest

from models.layers import (
    LayerNorm, CausalSelfAttention, MLP, Block,
    SwiGLU, CastedEmbedding, CastedSparseEmbedding, RotaryEmbedding,
    rms_norm,
)
from models.nanogpt import GPTConfig


def _base_config(**kw):
    cfg = GPTConfig(
        block_size=32, vocab_size=12, n_layer=2, n_head=2, n_embd=16,
        dropout=0.0, bias=False, activation='gelu', normalize='layernorm',
        pos_encodings='learned', forward_dtype=torch.float32,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ── LayerNorm ──────────────────────────────────────────────────────────────────

def test_layernorm_no_bias():
    ln = LayerNorm(16, bias=False)
    x = torch.randn(2, 8, 16)
    out = ln(x)
    assert out.shape == x.shape


def test_layernorm_with_bias():
    ln = LayerNorm(16, bias=True)
    x = torch.randn(2, 8, 16)
    out = ln(x)
    assert out.shape == x.shape


# ── rms_norm ───────────────────────────────────────────────────────────────────

def test_rms_norm_shape():
    x = torch.randn(2, 8, 16)
    out = rms_norm(x, variance_epsilon=1e-5)
    assert out.shape == x.shape


# ── CausalSelfAttention ────────────────────────────────────────────────────────

def test_attention_output_shape():
    cfg = _base_config()
    attn = CausalSelfAttention(cfg)
    B, T, C = 2, 8, cfg.n_embd
    x = torch.randn(B, T, C)
    out = attn(x)
    assert out.shape == (B, T, C)


def test_attention_with_rope():
    cfg = _base_config(pos_encodings='rotary')
    from models.layers import RotaryEmbedding
    B, T, C = 2, 8, cfg.n_embd
    rotary = RotaryEmbedding(dim=cfg.n_embd // cfg.n_head, max_position_embeddings=64, base=10000.0)
    cos_full, sin_full = rotary()
    # apply_rotary_pos_emb expects cos/sin sliced to the actual sequence length T
    cos_sin = (cos_full[:T], sin_full[:T])
    attn = CausalSelfAttention(cfg)
    x = torch.randn(B, T, C)
    out = attn(x, cos_sin=cos_sin)
    assert out.shape == (B, T, C)


# ── MLP ───────────────────────────────────────────────────────────────────────

def test_mlp_gelu():
    cfg = _base_config(activation='gelu')
    mlp = MLP(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    out = mlp(x)
    assert out.shape == x.shape


def test_mlp_relu():
    cfg = _base_config(activation='relu')
    mlp = MLP(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    out = mlp(x)
    assert out.shape == x.shape


# ── SwiGLU ────────────────────────────────────────────────────────────────────

def test_swiglu_shape():
    swiglu = SwiGLU(hidden_size=16, expansion=2.0)
    x = torch.randn(2, 8, 16)
    out = swiglu(x)
    assert out.shape == x.shape


# ── Block ─────────────────────────────────────────────────────────────────────

def test_block_layernorm():
    cfg = _base_config(normalize='layernorm')
    block = Block(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    out = block(x)
    assert out.shape == x.shape


def test_block_rmsnorm():
    cfg = _base_config(normalize='rmsnorm')
    block = Block(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    out = block(x)
    assert out.shape == x.shape


# ── CastedEmbedding ────────────────────────────────────────────────────────────

def test_casted_embedding_shape():
    emb = CastedEmbedding(num_embeddings=10, embedding_dim=16, init_std=0.02, cast_to=torch.float32)
    idx = torch.arange(10)
    # CastedEmbedding uses the weight directly (not a forward lookup)
    # The weight is shape (num_embeddings, embedding_dim)
    assert emb.embedding_weight.shape == (10, 16)


# ── CastedSparseEmbedding ──────────────────────────────────────────────────────

def test_sparse_embedding_train():
    emb = CastedSparseEmbedding(num_embeddings=20, embedding_dim=8, batch_size=4,
                                 init_std=0.01, cast_to=torch.float32)
    emb.train()
    ids = torch.randint(0, 20, (4,))
    out = emb(ids)
    assert out.shape == (4, 8)
    assert out.requires_grad


def test_sparse_embedding_eval():
    emb = CastedSparseEmbedding(num_embeddings=20, embedding_dim=8, batch_size=4,
                                 init_std=0.01, cast_to=torch.float32)
    emb.eval()
    ids = torch.randint(0, 20, (4,))
    with torch.no_grad():
        out = emb(ids)
    assert out.shape == (4, 8)


# ── RotaryEmbedding ────────────────────────────────────────────────────────────

def test_rotary_embedding_shapes():
    rope = RotaryEmbedding(dim=8, max_position_embeddings=32, base=10000.0)
    cos, sin = rope()
    assert cos.shape == (32, 8)
    assert sin.shape == (32, 8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
