
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
    model_name = model_config.get('model_type', 'nanogpt')
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model type: {model_name}")
    
    if model_name == 'nanogpt':
        model_config = nanogpt.GPTConfig(**model_config['config'])
    
    model_cls = model_map[model_name]
    model = model_cls(model_config)
    
    resume = model_config.get('resume', False)
    if resume:
        # TODO: Add logic to resume model training or loading here
        pass
    
    compile_model = model_config.get('compile', False)
    if compile_model:
        model = torch.compile(model)
    
    return model


def get_optimizer(config: dict,
                  model: Module) -> torch.optim.Optimizer:
    optim_config = config['OPTIMIZER']
    optim_name = optim_config.get('optimizer_type', 'adamw')
    
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
    
    return optimizer


def get_scheduler(config: dict,
                  optimizer: torch.optim.optimizer):
    scheduler_config = config['SCHEDULER']
    
    if 'SCHEDULER' not in config:
        return None
    
    sched_map = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'steplr': torch.optim.lr_scheduler.StepLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'reduceonplateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    
    scheduler_name = scheduler_config.get('scheduler_type', 'cosine')
    if scheduler_name not in sched_map:
        raise ValueError(f"Unknown scheduler type: {scheduler_name}")
    
    sched_cls = sched_map[scheduler_name]
    scheduler = sched_cls(optimizer, **scheduler_config['config'])
    
    return scheduler