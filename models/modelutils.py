"""
Include model and training configuration utilities.
"""
import os
import inspect
from typing import Union

import torch
from adam_atan2_pytorch import AdamAtan2

import models.nanogpt as nanogpt
from models.optimizer import CastedSparseEmbeddingSignSGD_Distributed, CombinedOptimizer
from models.scheduler import NanoGPTScheduler
from utils.logger import load_ckpt


def get_model(config: dict,
              device: str,
              num_identifiers: int=0,
              vocab_size: int=12,
              batch_size: int=32,
              ignore_label_id: int=-100) -> torch.nn.Module:
    model_config = config['model']
    
    init_from = config['logging'].get('init_from', 'scratch')
    if init_from == 'resume':
        checkpoint = load_ckpt(config, device='cpu')
        
        checkpoint_model_args = checkpoint['model_args']
        model_config['config'].update(checkpoint_model_args)
        
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model_type = model_config.get('type', 'nanogpt').lower()
    _model_config = model_config.get('config', {})
    
    if model_type == 'nanogpt':
        _model_config['num_identifiers'] = num_identifiers
        _model_config['vocab_size'] = vocab_size
        _model_config['batch_size'] = batch_size
        _model_config['ignore_label_id'] = ignore_label_id
        model_config = nanogpt.GPTConfig(**_model_config)
        model = nanogpt.GPT(model_config)
        if init_from == 'resume':
            model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    is_compile = config['model'].get('compile', False)
    if is_compile:
        model = torch.compile(model)
        
    model.to(device)
    
    return model


def get_optimizer(config: dict, 
                  model: torch.nn.Module,
                  device: torch.device) -> torch.optim.Optimizer:
    optimizer_config = config['training']['optimizer']
    
    optimizer_type = optimizer_config.get('type', 'adamw')
    
    optim_config = optimizer_config.get('config', {})
    if optimizer_type == 'adamw':
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        if use_fused:
            optim_config['fused'] = True
        optimizer = torch.optim.AdamW(model.parameters(), **optim_config)
    elif optimizer_type == 'adamaten2':
        puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            model.puzzle_emb.buffers(),
            lr=0,
            weight_decay=0.1,
            world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        )
        adam_optimizer = AdamAtan2(model.parameters(), **optim_config)
        optimizer = CombinedOptimizer(puzzle_emb_optimizer, adam_optimizer)
    elif optimizer_type == 'distributed':
        # TODO: implement distributed optimizer
        raise NotImplementedError("Distributed optimizer is not implemented yet.")
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
    init_from = config['logging'].get('init_from', 'scratch')
    if init_from == 'resume':
        checkpoint = load_ckpt(config, device='cpu')
        
        if optimizer_type == 'distributed':
            raise NotImplementedError("Distributed optimizer is not implemented yet.")
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
    
    return optimizer


def get_scheduler(config: dict, 
                  optimizer: torch.optim.Optimizer):
    scheduler_type = config['training']['scheduler'].get('type', 'compose').lower()
    
    if scheduler_type == 'compose':
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
    elif scheduler_type == 'hrm':
        scheduler_config = {
            'base_lr': config['training'].get('learning_rate', 1e-4),
            'num_warmup_steps': config['training'].get('warmup_iters', 2000),
            'num_training_steps': config['training'].get('max_iters', 600000),
            'min_ratio': config['training'].get('min_lr_ratio', 0.1),
        }
        from models.scheduler import CosineSchedulerWithWarmup
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            **scheduler_config
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    return scheduler