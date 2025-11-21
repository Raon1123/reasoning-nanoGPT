from dataclasses import dataclass
from typing import (
    Union
)

import torch
from torch import nn



@dataclass
class TrainConfigs:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: Union[torch.cuda.amp.GradScaler, None]  # GradScaler for mixed precision
    lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None]  # LR scheduler
    
    dataloader: torch.utils.data.DataLoader  # Iterator yielding (X, Y) batches
    grad_clip: float
    
    gradient_accumulation_steps: int
    ddp: bool
    device: str
    ctx: Union[torch.amp.autocast, torch.cuda.amp.autocast, None]

def train_epoch(config: TrainConfigs, num_steps: int) -> dict:
    """
    Perform training for num_steps steps, handling gradient accumulation and DDP.
    Returns a dict of metrics (e.g., average loss, current LR).
    """
    model = config.model
    optimizer = config.optimizer
    scaler = config.scaler
    lr_scheduler = config.lr_scheduler
    dataloader = config.dataloader
    grad_clip = config.grad_clip
    gradient_accumulation_steps = config.gradient_accumulation_steps
    ddp = config.ddp
    ctx = config.ctx
    
    model.train()
    total_loss = 0.0
    step_count = 0
    
    # Iterator for dataloader
    data_iter = iter(dataloader)
    
    for _ in range(num_steps):
        for micro_step in range(gradient_accumulation_steps):
            # DDP: Sync gradients only on the last micro-step
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
            try:
                X, Y = next(data_iter)
            except StopIteration:
                # If dataloader is finite, restart or break (assuming infinite for simplicity)
                data_iter = iter(dataloader)
                X, Y = next(data_iter)
            
            X, Y = X.to(config.device), Y.to(config.device)
            
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps  # Scale for accumulation
            
            # Backward with scaler
            scaler.scale(loss).backward()
            total_loss += loss.item()
            step_count += 1
        
        # Step optimizer after accumulation
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Step LR scheduler if present
        if lr_scheduler:
            lr_scheduler.step()
    
    avg_loss = total_loss / max(step_count, 1)  # Avoid div by zero
    current_lr = optimizer.param_groups[0]['lr']
    return {'loss': avg_loss, 'lr': current_lr}


@torch.no_grad()
def test_epoch(config: TrainConfigs, num_steps: int) -> dict:
    """
    Perform evaluation for num_steps steps.
    Returns a dict of metrics (e.g., average loss).
    """
    model = config.model
    dataloader = config.dataloader
    ctx = config.ctx
    
    model.eval()
    total_loss = 0.0
    
    # Iterator for dataloader
    data_iter = iter(dataloader)
    
    for _ in range(num_steps):
        try:
            X, Y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            X, Y = next(data_iter)
        
        X, Y = X.to(config.device), Y.to(config.device)
        
        with ctx:
            logits, loss = model(X, Y)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / max(num_steps, 1)
    return {'loss': avg_loss}
    
    
