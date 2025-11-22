import os
import time
import math
import pickle
from contextlib import nullcontext
from typing import Dict, Any, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import GPTConfig, GPT, TRM, HRM, NanoChat
from src.utils import set_seed, get_ctx
from data import get_batch, get_data, get_dataloader

def train(config: Dict[str, Any]) -> None:
    """
    Main training function.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    # Extract config
    out_dir = config['system']['out_dir']
    device = config['system']['device']
    dtype = config['system']['dtype']
    compile_model = config['system']['compile']
    backend = config['system']['backend']
    init_from = config['system']['init_from']
    
    dataset = config['data']['dataset']
    gradient_accumulation_steps = config['data']['gradient_accumulation_steps']
    batch_size = config['data']['batch_size']
    block_size = config['data']['block_size']
    
    learning_rate = float(config['optimizer']['learning_rate'])
    max_iters = config['optimizer']['max_iters']
    weight_decay = float(config['optimizer']['weight_decay'])
    beta1 = config['optimizer']['beta1']
    beta2 = config['optimizer']['beta2']
    grad_clip = config['optimizer']['grad_clip']
    decay_lr = config['optimizer']['decay_lr']
    warmup_iters = config['optimizer']['warmup_iters']
    lr_decay_iters = config['optimizer']['lr_decay_iters']
    min_lr = float(config['optimizer']['min_lr'])
    
    eval_interval = config['logging']['eval_interval']
    log_interval = config['logging']['log_interval']
    eval_iters = config['logging']['eval_iters']
    eval_only = config['logging']['eval_only']
    always_save_checkpoint = config['logging']['always_save_checkpoint']
    wandb_log = config['logging']['wandb_log']
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    ctx = get_ctx(device_type)
    
    # Model init
    iter_num = 0
    best_val_loss = 1e9
    
    model_type = config['model'].get('type', 'gpt')
    model_args = config['model']['config'].copy()
    
    # Add system/data args to model_args
    model_args['block_size'] = block_size
    model_args['vocab_size'] = None # Will be updated later

    # Data loader
    from data import get_data, get_dataloader
    
    train_dataset = get_data(config, split='train')
    val_dataset = get_data(config, split='val')
    
    # Attempt to get vocab size from dataset if possible, or meta.pkl
    # Since we moved data loading, we might need to check meta.pkl manually or add a method to dataset
    # For now, let's check meta.pkl in the data directory as before
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    if init_from == 'scratch':
        print(f"Initializing a new {model_type} model from scratch")
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        
        if model_type == 'gpt':
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        elif model_type == 'nanochat':
            # NanoChat might have same config structure for now
            gptconf = GPTConfig(**model_args)
            model = NanoChat(gptconf)
        elif model_type == 'trm':
            from models import TRMConfig
            trmconf = TRMConfig(**model_args)
            model = TRM(trmconf)
        elif model_type == 'hrm':
            from models import HRMConfig
            hrmconf = HRMConfig(**model_args)
            model = HRM(hrmconf)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        
        # Determine model type from config or checkpoint if saved
        # For now, assume config has the correct type or we default to GPT
        # Ideally checkpoint should save model_type
        
        if model_type == 'gpt':
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        elif model_type == 'nanochat':
            gptconf = GPTConfig(**model_args)
            model = NanoChat(gptconf)
        elif model_type == 'trm':
            from models import TRMConfig
            # Restore TRM specific args if in checkpoint, else from config
            model_args['n_recurrence'] = checkpoint_model_args.get('n_recurrence', config['model']['config'].get('n_recurrence', 1))
            trmconf = TRMConfig(**model_args)
            model = TRM(trmconf)
        elif model_type == 'hrm':
            from models import HRMConfig
            model_args['n_planner_layers'] = checkpoint_model_args.get('n_planner_layers', config['model']['config'].get('n_planner_layers', 2))
            model_args['n_actor_layers'] = checkpoint_model_args.get('n_actor_layers', config['model']['config'].get('n_actor_layers', 4))
            hrmconf = HRMConfig(**model_args)
            model = HRM(hrmconf)
        else:
             raise ValueError(f"Unknown model type: {model_type}")

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)

    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size
    
    model.to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    if compile_model:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    ddp_info = {
        'ddp': ddp,
        'rank': ddp_rank if ddp else 0,
        'world_size': ddp_world_size if ddp else 1
    }
    
    train_loader = get_dataloader(train_dataset, config, ddp_info)
    val_loader = get_dataloader(val_dataset, config, ddp_info)
    
    # Logging
    from src.logging import Logger
    logger = Logger(config, master_process=master_process)

    # Create iterators
    train_iter = iter(train_loader)

    # ... (previous code) ...

    # Define iteration functions
    def train_step(model: torch.nn.Module, batch: Any, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, gradient_accumulation_steps: int, ctx: Any, grad_clip: float) -> float:
        """
        Perform a single training step.

        Args:
            model (torch.nn.Module): The model.
            batch (Any): The input batch (X, Y).
            optimizer (torch.optim.Optimizer): The optimizer.
            scaler (torch.amp.GradScaler): Gradient scaler for AMP.
            gradient_accumulation_steps (int): Number of steps for gradient accumulation.
            ctx (Any): Autocast context.
            grad_clip (float): Gradient clipping value.

        Returns:
            float: The loss value.
        """
        X, Y = batch
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        
        # Forward pass
        with ctx:
            logits, loss, _ = model(X, Y)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        return loss.item() * gradient_accumulation_steps

    @torch.no_grad()
    def inference_iteration(model: torch.nn.Module, loader: torch.utils.data.DataLoader, eval_iters: int, ctx: Any, device: Any) -> float:
        """
        Run inference (evaluation) on a dataset loader.

        Args:
            model (torch.nn.Module): The model.
            loader (torch.utils.data.DataLoader): Data loader.
            eval_iters (int): Number of iterations to evaluate.
            ctx (Any): Autocast context.
            device (Any): Torch device.

        Returns:
            float: Mean loss.
        """
        model.eval()
        losses = torch.zeros(eval_iters)
        loader_iter = iter(loader)
        for k in range(eval_iters):
            try:
                X, Y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                X, Y = next(loader_iter)
            
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            
            with ctx:
                logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        model.train()
        return losses.mean()

    # ... (rest of setup) ...

    X, Y = next(train_iter) # Initial batch
    
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    
    # Scheduler
    if decay_lr:
        def lr_lambda(it: int) -> float:
            if it < warmup_iters:
                return (it + 1) / (warmup_iters + 1)
            if it > lr_decay_iters:
                return min_lr / learning_rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (min_lr + coeff * (learning_rate - min_lr)) / learning_rate
        
        # Ensure initial_lr is set for resumption compatibility
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = learning_rate

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=iter_num)
        
        # Manually set the initial LR for the current iter_num to avoid calling scheduler.step() before optimizer.step()
        # This prevents the PyTorch warning while ensuring correct warmup
        current_lr = learning_rate * lr_lambda(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    else:
        scheduler = None

    while iter_num < max_iters:
        # lr is managed by scheduler (updated at the end of step)
        lr = optimizer.param_groups[0]['lr']
        
        if iter_num % eval_interval == 0 and master_process:
            # Evaluation loop
            losses = {}
            for split, loader in [('train', train_loader), ('val', val_loader)]:
                losses[split] = inference_iteration(model, loader, eval_iters, ctx, device)
            
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            metrics = {
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }
            logger.log(metrics, step=iter_num)
            
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        if iter_num == 0 and eval_only:
            break

        # Training iteration
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
            # Use current batch X, Y
            # Note: train_step expects batch as (X, Y) tuple
            # But we need to handle the "fetch next batch" logic which was interleaved.
            # Let's encapsulate the forward/backward part in train_step, but data fetching might remain outside or be passed in.
            # To strictly follow "training iteration function", we could pass the iterator?
            
            # Let's refine train_step to take just the tensors for the computation part
            # and handle data loading here to keep the loop structure clear for grad accum.
            
            loss_val = train_step(model, (X, Y), optimizer, scaler, gradient_accumulation_steps, ctx, grad_clip)
            
            # Fetch next batch for next micro_step or next iteration
            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)
        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        if scheduler:
            scheduler.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # loss_val is from the last micro_step, which is a bit noisy but fine.
            # Actually loss_val returned by train_step is scaled back up.
            lossf = loss_val 
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        iter_num += 1
        local_iter_num += 1

    if ddp:
        destroy_process_group()
