"""
Tiny Recursive Model (TRM) — faithful re-implementation based on:
  "Less is More: Recursive Reasoning with Tiny Networks"
  SamsungSAILMontreal/TinyRecursiveModels

Key differences from a standard transformer:
  1. WEIGHT SHARING — a single L_level module (stack of Transformer blocks)
     is reused across ALL H_cycles and L_cycles iterations.
  2. TWO LATENT STATES — z_H (high-level) and z_L (low-level), each
     initialized from learnable buffers H_init / L_init.
  3. NO-GRAD OPTIMIZATION — the first (H_cycles-1) outer iterations run
     inside torch.no_grad() for memory efficiency; only the final iteration
     retains gradients.
  4. INJECTION — L_level(hidden, injection) adds injection before the
     block stack: hidden = hidden + injection; for blk in blocks: ...

Forward pass pseudocode:
    z_H = H_init.expand(B, T, n_embd)
    z_L = L_init.expand(B, T, n_embd)
    input_emb = embed(x)

    with no_grad:
        for h in range(H_cycles - 1):
            for l in range(L_cycles):
                z_L = L_level(z_L, z_H + input_emb)
            z_H = L_level(z_H, z_L)

    for l in range(L_cycles):         # final H step WITH grad
        z_L = L_level(z_L, z_H + input_emb)
    z_H = L_level(z_H, z_L)

    logits = lm_head(z_H)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import (
    Block,
    CastedSparseEmbedding,
    CastedEmbedding,
    LayerNorm,
    RotaryEmbedding,
)
from utils.toolkit import trunc_normal_init_


@dataclass
class TRMConfig:
    # Sequence / vocabulary
    block_size: int = 900
    vocab_size: int = 12
    ignore_label_id: int = -100

    # Core TRM hyperparameters
    H_cycles: int = 3       # outer recurrence depth
    L_cycles: int = 4       # inner recurrence depth per H step

    # Shared block hyperparameters (same fields as GPTConfig so Block works)
    n_layer: int = 4        # number of transformer layers inside L_level
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    activation: str = 'swiglu'
    expansion: float = 2.667        # for swiglu: keeps param count ≈ 4× gelu
    normalize: str = 'rmsnorm'
    pos_encodings: str = 'rotary'
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Sparsity (same as GPT)
    sparsity: float = 1.0

    # Puzzle embeddings
    puzzle_emb_ndim: int = 0
    num_identifiers: int = 0
    batch_size: int = 32

    # dtype for forward pass
    forward_dtype: torch.dtype = torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# Shared reasoning module (L_level)
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningModule(nn.Module):
    """
    A weight-shared stack of Transformer blocks.
    The same weights are reused at every (H, L) cycle.

    forward(hidden, injection):
        hidden = hidden + injection
        for blk in blocks:
            hidden = blk(hidden)
        return hidden
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        # n_layer shared blocks — all reused every cycle
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_out = (
            LayerNorm(config.n_embd, bias=config.bias)
            if config.normalize == 'layernorm'
            else None  # rmsnorm is applied inside Block already via ln_1/ln_2
        )

    def forward(self,
                hidden: torch.Tensor,
                injection: torch.Tensor,
                cos_sin=None) -> torch.Tensor:
        hidden = hidden + injection
        for blk in self.blocks:
            hidden = blk(hidden, cos_sin=cos_sin)
        return hidden


# ─────────────────────────────────────────────────────────────────────────────
# TRM model
# ─────────────────────────────────────────────────────────────────────────────

class TRM(nn.Module):
    """
    Tiny Recursive Model.
    Drop-in replacement for GPT with the same forward signature:
        forward(idx, puzzle_idx=None, targets=None, test_mode=False)
    """

    def __init__(self, config: TRMConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # ── Puzzle embeddings ─────────────────────────────────────────────
        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.n_embd)  # ceil div
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=config.num_identifiers,
                embedding_dim=self.puzzle_emb_len,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=config.forward_dtype,
            )

        # ── Token embedding ───────────────────────────────────────────────
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.embed_scale = math.sqrt(config.n_embd)

        # ── Positional encoding ───────────────────────────────────────────
        pos_enc = config.pos_encodings.lower()
        assert pos_enc in ('rotary', 'learned'), f"Unknown pos_encodings: {pos_enc}"

        total_pos = config.block_size + self.puzzle_emb_len
        if pos_enc == 'learned':
            embed_init_std = 1.0 / self.embed_scale
            self.emb_pos = CastedEmbedding(
                num_embeddings=total_pos,
                embedding_dim=config.n_embd,
                init_std=embed_init_std,
                cast_to=config.forward_dtype,
            )
        else:
            self.emb_pos = None
            self.rotary_emb = RotaryEmbedding(
                dim=config.n_embd // config.n_head,
                max_position_embeddings=total_pos,
                base=config.rope_theta,
            )

        # ── Weight-shared reasoning module ────────────────────────────────
        self.L_level = ReasoningModule(config)

        # ── LM head (weight-tied to wte) ──────────────────────────────────
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        # ── Learnable latent init vectors ─────────────────────────────────
        # Shape (n_embd,) — broadcast to (B, T, n_embd) at forward time.
        # Initialized with truncated normal (same as TRM paper).
        self.register_buffer(
            'H_init',
            trunc_normal_init_(torch.empty(config.n_embd, dtype=config.forward_dtype), std=1.0),
            persistent=True,
        )
        self.register_buffer(
            'L_init',
            trunc_normal_init_(torch.empty(config.n_embd, dtype=config.forward_dtype), std=1.0),
            persistent=True,
        )

        # ── Weight initialization ─────────────────────────────────────────
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TRM parameters: {n_params / 1e6:.2f}M  "
              f"(H_cycles={config.H_cycles}, L_cycles={config.L_cycles}, "
              f"n_layer={config.n_layer}, n_embd={config.n_embd})")

    # ── Utilities ─────────────────────────────────────────────────────────

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ── Input embeddings ─────────────────────────────────────────────────

    def _input_embeddings(self,
                          idx: torch.Tensor,
                          puzzle_idx: Optional[torch.Tensor]) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Sequence length {t} exceeds block_size {self.config.block_size}"

        pos = torch.arange(0, t + self.puzzle_emb_len, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)  # (b, t, n_embd)

        if self.config.puzzle_emb_ndim > 0:
            puzzle_emb = self.puzzle_emb(puzzle_idx)
            pad_count = self.puzzle_emb_len * self.config.n_embd - puzzle_emb.size(1)
            if pad_count > 0:
                puzzle_emb = F.pad(puzzle_emb, (0, pad_count), value=0.0)
            tok_emb = torch.cat(
                (puzzle_emb.view(-1, self.puzzle_emb_len, self.config.n_embd), tok_emb),
                dim=-2,
            )

        if self.config.pos_encodings == 'learned':
            pos_emb = self.emb_pos(pos)
            tok_emb = 0.7071067811865475 * (tok_emb + pos_emb)

        return self.embed_scale * tok_emb

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self,
                idx: torch.Tensor,
                puzzle_idx: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None,
                test_mode: bool = False):

        b, t = idx.size()
        seq_len = t + self.puzzle_emb_len

        cos_sin = None
        if hasattr(self, 'rotary_emb'):
            cos_full, sin_full = self.rotary_emb()
            cos_sin = (cos_full[:seq_len], sin_full[:seq_len])

        # Input embeddings: (b, seq_len, n_embd)
        input_emb = self._input_embeddings(idx, puzzle_idx)

        # Initialize latent states from learnable buffers
        z_H = self.H_init[None, None, :].expand(b, seq_len, -1).clone()
        z_L = self.L_init[None, None, :].expand(b, seq_len, -1).clone()

        # ── TRM recurrence ────────────────────────────────────────────────
        # First (H_cycles-1) outer iterations: no grad for memory efficiency
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
                    z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # Final outer iteration: WITH grad
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_emb, cos_sin=cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # ── Output ────────────────────────────────────────────────────────
        if targets is not None:
            logits = self.lm_head(z_H)

            # Strip puzzle-embedding prefix positions from logits
            if self.puzzle_emb_len > 0:
                logits = logits[:, self.puzzle_emb_len:, :]

            if self.config.sparsity < 1.0 and not test_mode:
                _, T = targets.size()
                mask_size = int(T * self.config.sparsity)
                rand_idx = torch.randperm(T, device=targets.device)[:mask_size]
                loss = F.cross_entropy(
                    logits[:, rand_idx, :].to(torch.float32).contiguous().view(-1, logits.size(-1)),
                    targets[:, rand_idx].to(torch.long).view(-1),
                    ignore_index=self.config.ignore_label_id,
                )
            else:
                loss = F.cross_entropy(
                    logits.to(torch.float32).contiguous().view(-1, logits.size(-1)),
                    targets.to(torch.long).view(-1),
                    ignore_index=self.config.ignore_label_id,
                )
        else:
            logits = self.lm_head(z_H[:, [-1], :])
            loss = None

        return logits, loss
