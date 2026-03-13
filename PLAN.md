# Refactoring & Ablation Plan

## 1. Bugs Found

### B1 — `epochs.py:30` — `pbar_iter` used before first assignment
**File**: `utils/epochs.py`
**Severity**: High (logic error, silently swallowed by bare `except`)
**Description**: Inside `eval_epoch()`, the loop body does `next(pbar_iter)` on the first iteration before `pbar_iter` is ever assigned. The bare `except:` catches the resulting `NameError` and creates the iterator there, so it *happens* to work but hides the real error and resets the iterator on every call.
**Fix**: Initialize `pbar_iter = iter(sample_dataloader)` before the loop.

### B2 — `main.py:52` — `val_loader` loads train split, not a validation split
**File**: `main.py`
**Severity**: Medium (evaluation always reports train-set metrics as "val")
**Description**: `val_loader` is created with `split='train'`. The original intent was probably a held-out validation subset or the train split used for monitoring. Either clarify this is intentional or use a proper val split.
**Fix** (doc-level for now): Add a comment or rename to `trn_eval_loader` to make it explicit.

### B3 — `main.py:31` — `autocast` uses `float32` (no-op)
**File**: `main.py`
**Severity**: Medium (wastes memory — model runs fp16 weights but no autocast benefit)
**Description**: `torch.autocast(device_type=device_type, dtype=torch.float32)` is a no-op. The model config specifies `dtype: float16`. Should read dtype from config.
**Fix**: Read dtype from `config['model'].get('dtype', 'float16')` and map to `torch.dtype`.

### B4 — `main.py:86` — `train_iter` used before assignment
**File**: `main.py`
**Severity**: Medium (same pattern as B1)
**Description**: First call to `next(train_iter)` raises `NameError` caught by bare `except:`, which silently creates the iterator. Hides potential bugs and complicates debugging.
**Fix**: Initialize `train_iter = iter(train_loader)` before the loop.

### B5 — `main.py:181` — `scheduler.get_last_lr()` called unconditionally
**File**: `main.py`
**Severity**: Medium (AttributeError if scheduler is `None` or a `CombinedScheduler`)
**Description**: `current_lr = scheduler.get_last_lr()[0]` is called at every `eval_interval`. If `scheduler` is `None` or a callable this will raise.
**Fix**: Guard with `if scheduler is not None and hasattr(scheduler, 'get_last_lr')`.

### B6 — `modelutils.py:128` — `rank` not in scope of `get_optimizer()`
**File**: `models/modelutils.py`
**Severity**: High (NameError when `init_from='resume'`)
**Description**: `optimizer.load_state_dict(checkpoint['optimizer'], strict=(rank == 0))` references `rank` which is not a parameter of `get_optimizer()`.
**Fix**: Add `rank: int = 0` parameter to `get_optimizer()` and pass it from `main.py`.

### B7 — `modelutils.py:168-173` — `hrm` scheduler reads wrong config keys
**File**: `models/modelutils.py`
**Severity**: High (wrong warmup/min_lr defaults always used for hrm scheduler)
**Description**: `get_scheduler()` for type `hrm` reads `warmup_iters` and `min_lr_ratio` from `config['training']` (top level), but the YAML places them under `training.scheduler.config`. `max_iters` is correctly at `training` level.
**Fix**: Read from `config['training']['scheduler']['config']`.

### B8 — `layers.py:147` — `trunc_normal_init_` duplicated
**File**: `models/layers.py` and `utils/toolkit.py`
**Severity**: Low (code duplication)
**Description**: Identical function exists in both files. `nanogpt.py` imports from `utils.toolkit`, `layers.py` defines its own copy.
**Fix**: Remove from `layers.py`, have `layers.py` import from `utils.toolkit`.

### B9 — `eval_epoch` mask not moved to device before use
**File**: `utils/epochs.py`
**Severity**: Medium (device mismatch in metric computation)
**Description**: `mask = (Y != IGNORE_LABEL_ID)` is computed on line 36 before `Y` is moved to device (line 49). The mask stays on CPU and subsequent `correct & mask` may fail or produce wrong results after logits/Y are moved to GPU and then back to CPU.
**Fix**: Compute `mask` after `Y = device_fn(Y)` and after `Y = detach_fn(Y)`, or compute on CPU after detach.

---

## 2. Refactoring

### R1 — Remove `trunc_normal_init_` duplication (links to B8)
Keep only in `utils/toolkit.py`, import into `models/layers.py`.

### R2 — `main.py`: extract training step into a function
The inner training loop (lines 139-176) is hard to test in isolation. Extract to `train_step(model, batch, optimizer, scheduler, ...)`.

### R3 — `main.py`: named variables for config extraction at top
Currently config values are read inline throughout `main()`. Extract to a config struct or local variables at the start of `main()` for clarity.

### R4 — `CombinedScheduler` should not inherit `LRScheduler`
`CombinedScheduler` inherits from `LRScheduler` but only wraps two child schedulers. The base class `__init__` mutates the child optimizer's LRs on construction. This is fragile. Refactor to a plain class with `step()` and `get_last_lr()`.

### R5 — `get_optimizer` in `modelutils.py`: separate model/puzzle param groups properly
Currently the `puzzle` optimizer type adds the puzzle_emb to a separate optimizer but also passes `model.parameters()` (which includes puzzle_emb) to AdamW. This means puzzle_emb is in both optimizers. Fix to exclude puzzle_emb from model_optimizer.

---

## 3. Test Strategy

### Tests to write (all in `tests/`):

| File | What to test |
|---|---|
| `test_scheduler.py` | (exists) NanoGPTScheduler warmup/decay/min_lr |
| `test_cosine_scheduler.py` | CosineSchedulerWithWarmup behavior |
| `test_combined_scheduler.py` | CombinedScheduler step/get_last_lr |
| `test_model_forward.py` | GPT forward pass: shapes, loss computation, sparsity mask |
| `test_model_config.py` | GPTConfig defaults and validation |
| `test_optimizer.py` | CombinedOptimizer state_dict/load_state_dict |
| `test_dataloader.py` | get_dataloader returns correct types and shapes |
| `test_eval_epoch.py` | eval_epoch runs without NameError, returns expected keys |
| `test_layers.py` | LayerNorm, RoPE, SwiGLU forward shapes |

### B10 — `nanogpt.py:248` — `emb_pos.embedding_weight` used instead of positional lookup
**File**: `models/nanogpt.py`
**Severity**: High (RuntimeError whenever T < block_size)
**Description**: `_input_embeddings` computes `pos = torch.arange(0, t+puzzle_emb_len)` but never uses it. Instead it adds `self.emb_pos.embedding_weight` (shape `block_size × n_embd`) directly, which only broadcasts correctly when `t == block_size`.
**Fix** (applied): Use `self.emb_pos(pos)` to look up the embeddings for the actual positions.

### B11 — `layers.py` — RoPE `cos/sin` cache not sliced to T before applying
**File**: `models/layers.py`
**Severity**: High (RuntimeError whenever T < block_size with RoPE)
**Description**: `CausalSelfAttention.forward` calls `apply_rotary_pos_emb(q, k, cos, sin)` where `cos/sin` are the full cached tensors from `RotaryEmbedding` (shape `max_seq × head_dim`). When `T < max_seq`, the element-wise multiply fails.
**Fix** (applied): Slice `cos[:T]`, `sin[:T]` before calling `apply_rotary_pos_emb`.

---

## 4. Ablation Experiment Queue

### Design
- All experiment configs live in `config/experiments/`
- A queue runner script `scripts/run_queue.py` reads a `queue.yaml` listing experiment config paths and runs them sequentially (or can be submitted to a job scheduler)
- Each config overrides only what changes vs. a base config

### Ablation Groups

#### Group A: Architecture — Standard Transformer vs. Tiny Recursive
```
A1: separate_reasoning=false  (baseline)
A2: separate_reasoning=true
```

#### Group B: Model Depth
```
B1: n_layer=4
B2: n_layer=8   (baseline)
B3: n_layer=16
B4: n_layer=32
```

#### Group C: Embedding Dimension
```
C1: n_embd=128
C2: n_embd=256  (baseline)
C3: n_embd=512
```

#### Group D: Activation Function
```
D1: activation=relu
D2: activation=gelu  (baseline)
D3: activation=swiglu
```

#### Group E: Normalization
```
E1: normalize=layernorm  (baseline)
E2: normalize=rmsnorm
```

#### Group F: Positional Embeddings
```
F1: pos_encodings=learned  (baseline)
F2: pos_encodings=rotary
```

#### Group G: Optimizer
```
G1: optimizer.type=adamw  (baseline)
G2: optimizer.type=puzzle  (with puzzle_emb)
```

#### Group H: Scheduler
```
H1: scheduler.type=compose
H2: scheduler.type=hrm
```

#### Group T: TRM Cycle Ablation (H_cycles × L_cycles)
Compares TRM with different outer/inner recurrence depths.
All configs share the same base architecture (n_layer=4, n_embd=256, swiglu, rmsnorm, rotary).

```
T1: H_cycles=1, L_cycles=4  — single pass, no recurrence (baseline equivalent)
T2: H_cycles=3, L_cycles=4  — paper default outer depth
T3: H_cycles=3, L_cycles=6  — paper default for Sudoku (increased inner depth)
```

Configs: `config/experiments/T1_trm_H1_L4.yaml`, `T2_trm_H3_L4.yaml`, `T3_trm_H3_L6.yaml`

**Key properties of TRM** (faithfully implemented in `models/trm.py`):
- Single weight-shared `L_level` (stack of n_layer blocks) reused every cycle
- Two latent states `z_H` (high-level) and `z_L` (low-level), each from learnable buffers
- First `H_cycles-1` outer iterations run under `torch.no_grad()` for memory efficiency
- Final outer iteration retains gradients for backprop
- Injection: `hidden = hidden + injection` before block stack
