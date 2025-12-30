"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from typing import (
    Optional, Union
)
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.layers import (
    LayerNorm,
    Block,
    CastedSparseEmbedding,
    CastedEmbedding,
    RotaryEmbedding
)
from utils.toolkit import trunc_normal_init_


# FIXED: I modify 
@dataclass
class GPTConfig:
    block_size: int = 900
    vocab_size: int = 12 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    sparsity: float = 0.0  # Added to support sparse loss
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    puzzle_emb_ndim: int = 0  # Added to support puzzle embeddings
    
    # for NLP
    activation: str = 'relu'  # Activation function for MLP: 'relu', 'gelu', 'swiglu' etc.
    expansion: float = 4.0  # Expansion factor for MLP hidden dimension where swiglu is used
    normalize: str = 'layernorm'  # Normalization type: 'layernorm', 'rmsnorm', etc.
    pos_encodings: str = 'learned'  # Positional embedding type: 'rotary', 'learned', etc.
    
    rms_norm_eps: float = 1.0e-5  # Epsilon for RMSNorm
    rope_theta: float = 10000.0  # Rotary position embedding base theta
    
    num_identifiers: int = 0  # Added to support puzzle embeddings
    ignore_label_id: int = -100  # Added to support sparse loss
    batch_size: int = 32  # Added to support puzzle embeddings

class GPT(nn.Module):

    def __init__(self, 
                 config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.n_embd)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=self.config.num_identifiers,
                embedding_dim=self.puzzle_emb_len,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=torch.float32
            )

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # LM blocks
        pos_encodings = config.pos_encodings.lower()
        assert pos_encodings in ['rotary', 'learned'], \
            f"Unknown pos_encodings {pos_encodings}"
        
        self.embed_scale = math.sqrt(config.n_embd)
        if pos_encodings == 'learned':
            
            embed_init_std = 1.0 / self.embed_scale
            self.emb_pos = CastedEmbedding(
                num_embeddings=config.block_size + self.puzzle_emb_len,
                embedding_dim=config.n_embd,
                init_std=embed_init_std,
                cast_to=torch.float32
            )
        elif pos_encodings == 'rotary':
            self.emb_pos = None  # no learned positional embeddings needed
            self.rotary_emb = RotaryEmbedding(
                dim=config.n_embd // config.n_head,
                max_position_embeddings=config.block_size + self.puzzle_emb_len,
                base=config.rope_theta
            )
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                idx: torch.Tensor,
                puzzle_idx: Union[torch.Tensor, None] = None,
                targets: Union[torch.Tensor, None] = None,
                test_mode: bool = False) -> Union[torch.Tensor, Union[torch.Tensor, None]]:
        sequence_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, 'rotary_emb') else None
        )
        
        x = self._input_embeddings(idx, puzzle_idx)
        
        for block in self.transformer.h:
            x = block(x, cos_sin=sequence_info['cos_sin'])
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # range of logits and targets is (B, T, C) and (B, T) respectively
            if self.puzzle_emb_len > 0:
                # remove puzzle embedding positions from logits
                logits = logits[:, self.puzzle_emb_len:, :]
            
            if self.config.sparsity < 1.0 and not test_mode:
                # randomly mask out some targets for sparse loss computation
                _, T = targets.size()
                mask_size = int(T * self.config.sparsity)
                rand_indices = torch.randperm(T, device=targets.device)[:mask_size]
                sparse_targets = targets[:, rand_indices]
                sparse_logits = logits[:, rand_indices, :]
                
                loss = F.cross_entropy(sparse_logits.to(torch.float32).contiguous().view(-1, logits.size(-1)), sparse_targets.to(torch.long).view(-1), ignore_index=self.config.ignore_label_id).squeeze(-1)
            else:
                loss = F.cross_entropy(logits.to(torch.float32).contiguous().view(-1, logits.size(-1)), targets.to(torch.long).view(-1), ignore_index=self.config.ignore_label_id).squeeze(-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def _input_embeddings(self,
                          idx: torch.Tensor,
                          puzzle_idx: Union[torch.Tensor, None]) -> torch.Tensor:
        device = idx.device

        # normal embeddings here
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t+self.puzzle_emb_len, dtype=torch.long, device=device) # shape (t)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        # puzzle embeddings handle here
        if self.config.puzzle_emb_ndim > 0:
            puzzle_emb = self.puzzle_emb(puzzle_idx)  # shape (b, puzzle_emb_len)
            
            pad_count = self.puzzle_emb_len * self.config.n_embd - puzzle_emb.size(1)
            if pad_count > 0:
                puzzle_emb = F.pad(puzzle_emb, (0, pad_count), value=0.0)
                
            # concat to tkn emb
            tok_emb = torch.cat(
                (puzzle_emb.view(-1, self.puzzle_emb_len, self.config.n_embd), tok_emb),
                dim=-2
            )
                
        
        if self.config.pos_encodings == 'learned':
            # scale by inverse root square 2 to maintain forward variance
            tok_emb = 0.7071067811865475 * (tok_emb + self.emb_pos.embedding_weight.to(torch.float32))
        
        x = self.embed_scale * tok_emb
        
        return x

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    

