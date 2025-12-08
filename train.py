"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from utils.toolkit import load_yaml, load_json

from models.scheduler import NanoGPTScheduler as CosineWarmupScheduler
from utils.const import IGNORE_LABEL_ID

# -----------------------------------------------------------------------------
# I modify these parts frequently, so I manage configs by yaml files and command line args

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('--config', '-c', type=str, default='config/config_default.yaml',
                        help='Path to a config file (YAML) that specifies training parameters')
    args = parser.parse_args()
    return args

args = get_args()
config = load_yaml(args.config) if args.config is not None else {}

logging_config = config.get('logging', {})
data_config = config.get('dataset', {})
model_config = config.get('model', {})
training_config = config.get('training', {})

# I/O
out_dir = logging_config.get('output_dir', './out')
eval_interval = logging_config.get('eval_interval', 2000)
log_interval = logging_config.get('log_interval', 1)
eval_iters = logging_config.get('eval_iters', 200)
eval_only = logging_config.get('eval_only', False) # if True, script exits right after the first eval
always_save_checkpoint = logging_config.get('always_save_checkpoint', True) # if True, always save a checkpoint after each eval
init_from = logging_config.get('init_from', 'scratch') # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = logging_config.get('wandb_log', False) # disabled by default
wandb_project = logging_config.get('wandb_project', 'owt')
wandb_run_name = logging_config.get('wandb_run_name', 'gpt2') # 'run' + str(time.time())
# data
dataset = data_config.get('dataset', 'openwebtext')
gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 5 * 8) # used to simulate larger batch sizes
batch_size = training_config.get('batch_size', 12) # TODO: fix it for manage by global batch_size if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = training_config.get('block_size', 512)
# model
# Note that model config under model config
# adamw optimizer
learning_rate = training_config.get('optimizer', {}).get('config', {}).get('lr', 6e-4) # max learning rate
max_iters = training_config.get('max_iters', 600000) # total number of training iterations
grad_clip = training_config.get('grad_clip', 1.0) # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
scheduler_config = training_config.get('scheduler', {})
warmup_iters = scheduler_config.get('warmup_iters', 2000)
lr_decay_iters = scheduler_config.get('lr_decay_iters', 600000)
min_lr = scheduler_config.get('min_lr', 6e-5)
decay_lr = training_config.get('decay_lr', True)
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = model_config.get('dtype', 'float16')
compile = model_config.get('compile', True) # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', 'arc_agi', 'processed_data')

trn_root = os.path.join(data_dir, 'train')
tst_root = os.path.join(data_dir, 'test')

trn_X = np.load(os.path.join(trn_root, 'all__inputs.npy'), mmap_mode='r')
trn_y = np.load(os.path.join(trn_root, 'all__labels.npy'), mmap_mode='r')
trn_puzzle_identifiers = np.load(os.path.join(trn_root, 'all__puzzle_identifiers.npy'), mmap_mode='r') # actual puzzle IDs
trn_puzzle_indicies = np.load(os.path.join(trn_root, 'all__puzzle_indices.npy'), mmap_mode='r') # start/end indices for each puzzle in the dataset
lengths = trn_puzzle_indicies[1:] - trn_puzzle_indicies[:-1]
indices = np.arange(len(lengths))
trn_puzzle_indexes = np.repeat(indices, lengths)

trn_meta = load_json(os.path.join(trn_root, 'dataset.json'))
meta_vocab_size = trn_meta.get('vocab_size', 12)
ignore_label_id = trn_meta.get('ignore_label_id', IGNORE_LABEL_ID)

tst_X = np.load(os.path.join(tst_root, 'all__inputs.npy'), mmap_mode='r')
tst_y = np.load(os.path.join(tst_root, 'all__labels.npy'), mmap_mode='r')
tst_puzzle_identifiers = np.load(os.path.join(tst_root, 'all__puzzle_identifiers.npy'), mmap_mode='r')
tst_puzzle_indicies = np.load(os.path.join(tst_root, 'all__puzzle_indices.npy'), mmap_mode='r')
lengths = tst_puzzle_indicies[1:] - tst_puzzle_indicies[:-1]
indices = np.arange(len(lengths))
tst_puzzle_indexes = np.repeat(indices, lengths)

tst_meta = load_json(os.path.join(tst_root, 'dataset.json'))
assert meta_vocab_size == tst_meta.get('vocab_size', 12)
assert ignore_label_id == tst_meta.get('ignore_label_id', IGNORE_LABEL_ID)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        
    if split == 'train':
        ix = torch.randint(len(trn_X), (batch_size,))
        x = torch.from_numpy(trn_X[ix]).to(torch.long)
        y = torch.from_numpy(trn_y[ix]).to(torch.long)
        y[y == ignore_label_id] = IGNORE_LABEL_ID
        puzzle_ids = torch.from_numpy(trn_puzzle_indexes[ix]).to(torch.long)
    else:
        ix = torch.randint(len(tst_X), (batch_size,))
        x = torch.from_numpy(tst_X[ix]).to(torch.long)
        y = torch.from_numpy(tst_y[ix]).to(torch.long)
        y[y == ignore_label_id] = IGNORE_LABEL_ID
        puzzle_ids = torch.from_numpy(tst_puzzle_indexes[ix]).to(torch.long)
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        puzzle_ids = puzzle_ids.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        puzzle_ids = puzzle_ids.to(device)
    return x, y, puzzle_ids

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset


# model init
#model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
#                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
model_args = model_config.get('config', {})
model_args['block_size'] = block_size
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 12
model_args['ignore_label_id'] = ignore_label_id if model_config.get('ignore_idx', False) else IGNORE_LABEL_ID

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer_config = training_config.get('optimizer', {}).get('config', {})
optimizer_config['device_type'] = device_type
# TODO: I would like to add more optimizer such as Adam_Aten2 etc.
optimizer = model.configure_optimizers(**optimizer_config)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        valid_puzzle_counts = 0
        pixel_accuracies = torch.zeros(eval_iters)
        exact_corrects = 0
        
        for k in range(eval_iters):
            X, Y, puzzle_id = get_batch(split)
            # I should not understand why with ctx here causes some problem
            # min max of Y
            with torch.inference_mode():
                logits, loss = model(X, puzzle_id, Y, test_mode=True)
                
                # for calculate accuracy
                mask = (Y != IGNORE_LABEL_ID)
                preds = torch.argmax(logits, dim=-1)
                
                # correct per token (pixel)
                correct = (preds == Y) & mask
                accuracy = correct.sum().item() / mask.sum().item()
                
                # correct per all of sequence
                sequence_is_correct = correct.sum(-1) == mask.sum(-1)
            losses[k] = loss.item()
            pixel_accuracies[k] = accuracy
            exact_corrects += sequence_is_correct.sum().item()
            valid_puzzle_counts += X.size(0)
            
        out[f'{split}/loss'] = losses.mean()
        out[f'{split}/accuracy'] = pixel_accuracies.mean().item()
        out[f'{split}/sequence_accuracy'] = exact_corrects / valid_puzzle_counts
    model.train()
    return out

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, puzzle_id = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
tqdm_range = range(max_iters)
if master_process:
    from tqdm import tqdm
    tqdm_range = tqdm(tqdm_range, initial=iter_num, total=max_iters)

for epoch in tqdm_range:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    lr = optimizer.param_groups[0]['lr']

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: " + ", ".join([f"{k} {v:.4f}" for k,v in losses.items()]+[f"lr {lr:.6f}"]))
        if wandb_log:
            # append losses
            wandb_log = {
                "iter": iter_num,
                "lr": lr,
            }
            wandb_log.update(losses)

            wandb.log(wandb_log)
        if losses['test/loss'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['test/loss']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    #'scheduler': scheduler.state_dict() if decay_lr else None,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, puzzle_id, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, puzzle_id = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        #print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if isinstance(tqdm_range, tqdm):
            tqdm_range.set_description(f"loss {lossf:.4f}, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()
