"""
Include model and training configuration utilities.
"""
import os

import torch
from typing import Union

import models.nanochat as nanochat
import models.nanogpt as nanogpt
from models.scheduler import NanoGPTScheduler

def get_model(config: dict,
              device: str) -> torch.nn.Module:
    model_config = config['model']
    
    init_from = config['logging'].get('init_from', 'scratch')
    if init_from == 'resume':
        out_dir = config['logging'].get('output_dir', 'outs')
        run_name = config['logging'].get('run_name', 'default_run')
        print(f"Resuming from {out_dir}/{run_name}")
        
        ckpt_path = os.path.join(out_dir, run_name, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        checkpoint_model_args = checkpoint['model_args']
        model_config['config'].update(checkpoint_model_args)
        
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model_type = model_config.get('type', 'nanogpt')
    if model_type == 'nanogpt':
        model_config = nanogpt.GPTConfig(**model_config['config'])
        model = nanogpt.GPT(model_config)
        if init_from == 'resume':
            model.load_state_dict(state_dict)
    elif model_type == 'nanochat':
        model_config = nanochat.GPTConfig(**model_config['config'])
        model = nanochat.GPT(model_config)
        model.to_empty(device=device)
        model.init_weights()
        if init_from == 'resume':
            model.load_state_dict(state_dict)
        
    is_compile = config['model'].get('compile', False)
    if is_compile:
        model = torch.compile(model)
        
    model.to(device)
    
    return model


def get_optimizer(config: dict, model: torch.nn.Module) -> Union[torch.optim.Optimizer, tuple[torch.optim.Optimizer, torch.optim.Optimizer]]:
    optimizer_config = config['training']['optimizer']
    
    optimizer_type = optimizer_config.get('type', 'adamw')
    
    optim_config = optimizer_config.get('config', {})
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **optim_config)
    elif optimizer_type == 'distributed':
        unembedding_lr = optim_config.get('unembedding_lr', 0.004)
        embedding_lr = optim_config.get('embedding_lr', 0.2)
        matrix_lr = optim_config.get('matrix_lr', 0.02)
        weight_decay = optim_config.get('weight_decay', 0.0)
        optimizer = model.setup_optimizers(
            unembedding_lr=unembedding_lr, 
            embedding_lr=embedding_lr, 
            matrix_lr=matrix_lr, 
            weight_decay=weight_decay)
        adamw_optimizer, muon_optimizer = optimizer
        
    init_from = config['logging'].get('init_from', 'scratch')
    if init_from == 'resume':
        out_dir = config['logging'].get('output_dir', 'outs')
        run_name = config['logging'].get('run_name', 'default_run')
        print(f"Resuming optimizer from {out_dir}/{run_name}")
        
        ckpt_path = os.path.join(out_dir, run_name, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if optimizer_type == 'distributed':
            adamw_optimizer.load_state_dict(checkpoint['optimizer_adamw'])
            muon_optimizer.load_state_dict(checkpoint['optimizer_muon'])
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
    
    return optimizer


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