"""
Include model and training configuration utilities.
"""

import torch

from models.scheduler import NanoGPTScheduler

def get_model(config: dict) -> torch.nn.Module:
    pass


def get_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    pass


def get_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    scheduler_config = config['training']['scheduler']['config']
    warmup_iters = scheduler_config.get('warmup_iters', 2000)
    lr_decay_iters = scheduler_config.get('lr_decay_iters', 10000)
    min_lr = scheduler_config.get('min_lr', 6e-5)
    learning_rate = scheduler_config.get('learning_rate', 6e-4)
    scheduler = NanoGPTScheduler(
        optimizer,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=min_lr,
        max_lr=learning_rate
    )
    return scheduler