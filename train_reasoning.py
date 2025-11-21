import argparse
import os
import yaml
from contextlib import nullcontext

import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from data.datautils import (
    get_puzzle_dataloader,
)
from models.modelutil import (
    get_model,
    get_optimizer,
    get_scheduler
)
from src import toolkits
from src.epochs import train_epoch, test_epoch, TrainConfigs

def get_args():
    parser = argparse.ArgumentParser(description="Train a reasoning model using nanoGPT.")
    parser.add_argument('--config', type=str, required=True, help='Path to the training configuration file (YAML format).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda", "cpu").')
    args = parser.parse_args()
    return args


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = get_args()
    config = load_config(args.config)
    
    device = args.device
    
    ddp, device, master_process, seed_offset, gradient_accumulation_steps, tokens_per_iter, ddp_world_size = toolkits.setup_ddp(config, device)
    
    seed = config.get('SEED', 42)
    toolkits.fix_seeds(seed, seed_offset)
    
    # Note that we use only float32 or float16 **DO NOT** use bfloat16!!.
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    else:
        ctx = nullcontext()
        
    ddp_local_rank = int(os.environ['LOCAL_RANK']) if ddp else 0
    trn_dataloader = get_puzzle_dataloader(config, split='train', seed=seed, rank=ddp_local_rank, world_size=ddp_world_size)
    eval_dataloader = get_puzzle_dataloader(config, split='test', seed=seed, rank=ddp_local_rank, world_size=ddp_world_size)
    
    model = get_model(config)
    optimizer = get_optimizer(config, model)
    lr_scheduler = get_scheduler(config, optimizer)
    scaler = None  # scaler is not compatible
    
    grad_clip = config.get('TRAINING', {}).get('grad_clip', 1.0)
    
    model.to(device)
    
    model_compile = config['MODEL'].get('compile', False)
    if model_compile:
        print("Compiling model...")
        model = torch.compile(model)
        
    # distributed data parallel
    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank] if 'cuda' in device else None,
            output_device=ddp_local_rank if 'cuda' in device else None,
            find_unused_parameters=False,
        )
    
    # Create TrainConfigs
    trn_config = TrainConfigs(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        dataloader=trn_dataloader,
        grad_clip=grad_clip,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ddp=ddp,
        device=device,
        ctx=ctx
    )
    
    eval_config = TrainConfigs(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=None,  # No LR scheduling during eval
        dataloader=eval_dataloader,
        grad_clip=grad_clip,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ddp=ddp,
        device=device,
        ctx=ctx
    )
        
    # Training loop
    epochs = config.get('TRAINING', {}).get('epochs', 1)
    steps_per_epoch = config.get('TRAINING', {}).get('steps_per_epoch', 1000)
    eval_steps = config.get('TRAINING', {}).get('eval_steps', 200)
    
    for epoch in range(epochs):
        if master_process:
            print(f"Starting epoch {epoch + 1}/{epochs}...")
        
        # Train for steps_per_epoch
        train_metrics = train_epoch(trn_config, steps_per_epoch)
        
        # Evaluate
        test_metrics = test_epoch(eval_config, eval_steps)
        
        if master_process:
            print(f"Epoch {epoch + 1}: Train Loss {train_metrics['loss']:.4f}, Test Loss {test_metrics['loss']:.4f}, LR {train_metrics['lr']:.6f}")
        
    if ddp:
        destroy_process_group()
        
    
    
if __name__ == "__main__":
    main()