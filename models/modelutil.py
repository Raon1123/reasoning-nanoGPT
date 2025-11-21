
import torch
from torch.nn import Module

import models.nanogpt as nanogpt
import models.nanochat as nanochat

def get_model(config: dict) -> Module:
    
    # TODO: add more models here, such as HRM or TRM
    model_map = {
        'nanogpt': nanogpt.GPT,
        'nanochat': nanochat.GPT,
    }
    
    model_config = config['MODEL']
    model_name = model_config.get('model_type', 'nanogpt').lower()
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model type: {model_name}")
    
    if model_name == 'nanogpt':
        model_config = nanogpt.GPTConfig(**model_config['config'])
    
    model_cls = model_map[model_name]
    model = model_cls(model_config)
    
    if config['MODEL'].get('resume', False):
        model.load_state_dict(config['MODEL']['state_dict'])
    
    return model


def get_optimizer(config: dict,
                  model: Module) -> torch.optim.Optimizer:
    optim_config = config['OPTIMIZER']
    optim_name = optim_config.get('optimizer_type', 'adamw').lower()
    
    optim_map = {
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
    }
    
    if optim_name not in optim_map:
        raise ValueError(f"Unknown optimizer type: {optim_name}")
    
    optim_cls = optim_map[optim_name]
    optimizer = optim_cls(model.parameters(),
                          **optim_config['config'])
    
    if optim_config.get('resume', False):
        optimizer.load_state_dict(optim_config['state_dict'])
    
    return optimizer


def get_scheduler(config: dict,
                  optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler_config = config['SCHEDULER']
    
    if 'SCHEDULER' not in config:
        return None
    
    sched_map = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'steplr': torch.optim.lr_scheduler.StepLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'reduceonplateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'compose': get_default_scheduler,
    }
    
    scheduler_name = scheduler_config.get('scheduler_type', 'cosine')
    if scheduler_name not in sched_map:
        raise ValueError(f"Unknown scheduler type: {scheduler_name}")
    
    sched_cls = sched_map[scheduler_name]
    scheduler = sched_cls(optimizer, **scheduler_config['config'])
    
    return scheduler


def get_default_scheduler(optimizer: torch.optim.Optimizer,
                          **kwargs):
    warmup_iters = kwargs.get('warmup_iters', 2000)
    lr_decay_iters = kwargs.get('lr_decay_iters', 60000)
    min_lr = kwargs.get('min_lr', 6e-5)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            # Warmup: linear increase from 0 to learning_rate over warmup_iters
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=6.0e-4, end_factor=1.0, total_iters=warmup_iters),
            # Decay: cosine annealing from learning_rate to min_lr over (lr_decay_iters - warmup_iters)
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_decay_iters - warmup_iters, eta_min=min_lr)
    ],
    milestones=[warmup_iters]  # Switch at warmup_iters
    )
    
    return scheduler