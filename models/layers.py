from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.toolkit import trunc_normal_init_

CosSin = Tuple[torch.Tensor, torch.Tensor]

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, cos_sin: Optional[CosSin] = None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # RoPE — slice to actual T so the cache can be larger than the sequence
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos[:T], sin[:T])

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        activation_func = config.activation.lower()
        if activation_func == 'gelu':
            self.activation = nn.GELU()
        elif activation_func == 'relu':
            self.activation = nn.ReLU()
        elif activation_func == 'swiglu':
            self.activation = SwiGLU(config.n_embd, expansion=config.expansion)
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        normalze_func = config.normalize.lower()
        if normalze_func == 'layernorm':
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        elif normalze_func == 'rmsnorm':
            self.ln_1 = lambda x: rms_norm(x, variance_epsilon=config.rms_norm_eps)
            self.ln_2 = lambda x: rms_norm(x, variance_epsilon=config.rms_norm_eps)
        else:
            raise ValueError(f"Unsupported normalization: {config.normalize}")
        
        self.attn = CausalSelfAttention(config)
        
        if config.activation.lower() == 'swiglu':
            self.mlp = SwiGLU(config.n_embd, expansion=config.expansion)
        else:
            self.mlp = MLP(config)

    def forward(self, x, cos_sin: Optional[CosSin] = None):
        x = x + self.attn(self.ln_1(x), cos_sin=cos_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim] expect
    # q, k: [bs, num_heads, seq_len, head_dim] actually
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    #q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    #k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    q_embed = (q * cos.unsqueeze(-3)) + (rotate_half(q) * sin.unsqueeze(-3))
    k_embed = (k * cos.unsqueeze(-3)) + (rotate_half(k) * sin.unsqueeze(-3))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, bias=self.bias)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), dtype=cast_to), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq) # shape: 

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached
    
    
class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Real weights (updated with a custom optimizer, no autograd)
        self.weights = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), dtype=cast_to), std=init_std),
            requires_grad=False,
        )
        # Local embeddings used during the forward pass and optimized by our custom SignSGD optimizer
        self.local_weights = nn.Parameter(torch.zeros(batch_size, embedding_dim, dtype=cast_to), requires_grad=True)
        # Local IDs are needed by the optimizer but never trained
        self.local_ids = nn.Parameter(torch.zeros(batch_size, dtype=torch.long), requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                self.local_weights.copy_(self.weights[inputs])
                self.local_ids.copy_(inputs)
            return self.local_weights

        with torch.no_grad():
            return self.weights[inputs]
    
    
class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)