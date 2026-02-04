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
from models.scheduler import NanoGPTScheduler, CombinedScheduler
from utils.logger import load_ckpt


def get_model(config: dict,
              device: str,
              num_identifiers: int=0,
              vocab_size: int=12,
              batch_size: int=32,
              ignore_label_id: int=-100,
              world_size: int=1,
              rank: int=0) -> torch.nn.Module:
    model_config = config['model']
    
    init_from = config['logging'].get('init_from', 'scratch')
    if init_from == 'resume' and rank == 0:
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
        if init_from == 'resume' and rank == 0:
            model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # puzzle_params = model.puzzle_emb.buffers() 
    # print("Puzzle optimizer created for non-compiled model.") 
    # for param in puzzle_params:
    #     print(f"Puzzle param shape: {param.shape}, dtype: {param.dtype}, device: {param.device}, requires_grad: {param.requires_grad}, is_leaf: {param.is_leaf}")
        
    model.to(device)
    
    # Fix is_leaf=False issue after to(device) for puzzle parameters
    puzzle_params = model.puzzle_emb.buffers() 
    for param in puzzle_params:
        if param.requires_grad and not param.is_leaf:
            param.detach_()
            param.requires_grad = True
    
    is_compile = config['model'].get('compile', False)
    if is_compile:
        # must torch compile preserve the leaf status of puzzle embedding parameters?
        model = torch.compile(model,
                              fullgraph=True,
                              )
    
    
    if is_compile:
        if hasattr(model, '_orig_mod'):
             puzzle_params = model._orig_mod.puzzle_emb.buffers()
        elif hasattr(model, 'puzzle_emb'):
             puzzle_params = model.puzzle_emb.buffers()
        else:
             puzzle_params = []
    else:
        puzzle_params = model.puzzle_emb.buffers()
    
    with torch.device("cuda") if device.startswith("cuda") else torch.device("cpu"):
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    torch.distributed.broadcast(param, src=0)
    
    return model


def get_optimizer(config: dict, 
                  model: torch.nn.Module,
                  device: torch.device,
                  world_size: int=1,
                  rank: int=0) -> torch.optim.Optimizer:
    optimizer_config = config['training']['optimizer']
    
    optimizer_type = optimizer_config.get('type', 'adamw').lower()
    
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
            lr=1e-6,
            weight_decay=0.1,
            world_size=world_size
        )
        adam_optimizer = AdamAtan2(model.parameters(), **optim_config)
        optimizer = CombinedOptimizer([puzzle_emb_optimizer, adam_optimizer])
    elif optimizer_type == 'puzzle':
        print("We are using Puzzle optimizer.")
        puzzle_config = optimizer_config.get('puzzle_config', {})
        assert 'puzzle_lr' in puzzle_config, "puzzle_lr must be specified in puzzle_config"
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        if use_fused:
            optim_config['fused'] = True
        
        # if model is compiled, we cannot access model.puzzle_emb directly
        # if model is compiled, we cannot access model.puzzle_emb directly
        # check for _orig_mod (torch.compile)
        if hasattr(model, '_orig_mod'):
             puzzle_params = model._orig_mod.puzzle_emb.buffers()
        elif hasattr(model, 'puzzle_emb'):
            puzzle_params = model.puzzle_emb.buffers() 
            print(f"Device type: {device.type}, device number: {device.index}")
            for param in puzzle_params:
                print(f"Puzzle param shape: {param.shape}, dtype: {param.dtype}, device: {param.device}, requires_grad: {param.requires_grad}, is_leaf: {param.is_leaf}")
        else:
            puzzle_params = model.module.puzzle_emb.buffers()
        
        puzzle_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            puzzle_params, 
            lr=puzzle_config.get('puzzle_lr', 1e-3),
            weight_decay=puzzle_config.get('weight_decay', 0.1),
            world_size=world_size
        )
        model_optimizer = torch.optim.AdamW(
            model.parameters(),
            **optim_config
        )
        optimizer = CombinedOptimizer([puzzle_optimizer, model_optimizer])
        print("Puzzle optimizer created.")
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    assert optimizer is not None
        
    init_from = config['logging'].get('init_from', 'scratch')
    if init_from == 'resume':
        checkpoint = load_ckpt(config, device='cpu')
        
        if optimizer_type == 'distributed':
            raise NotImplementedError("Distributed optimizer is not implemented yet.")
        else:
            optimizer.load_state_dict(checkpoint['optimizer'], strict=(rank == 0))
    
    return optimizer


def get_scheduler(config: dict, 
                  optimizer: torch.optim.Optimizer,
                  last_epoch=-1) -> Union[torch.optim.lr_scheduler.LRScheduler, None]:
    scheduler_type = config['training']['scheduler'].get('type', 'compose').lower()
    
    if isinstance(optimizer, CombinedOptimizer):
        schedulers = []
        for opt in optimizer.optimizers:
            scheduler = get_scheduler(config, opt, last_epoch=last_epoch)
            schedulers.append(scheduler)
        return CombinedScheduler(schedulers, last_epoch=last_epoch)
            
        
    # base_lr from optimizer
    learning_rate = None
    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']
        break
    
    assert learning_rate is not None, "Failed to get learning rate from optimizer."
    
    if scheduler_type == 'compose':
        scheduler_config = config['training']['scheduler']['config']
        warmup_iters = scheduler_config.get('warmup_iters', 2000)
        lr_decay_iters = scheduler_config.get('lr_decay_iters', 10000)
        min_lr = scheduler_config.get('min_lr', 6e-5)
        scheduler = NanoGPTScheduler(
            optimizer,
            warmup_iters=warmup_iters,
            lr_decay_iters=lr_decay_iters,
            min_lr=min_lr,
            max_lr=learning_rate
        )
    elif scheduler_type == 'hrm':
        scheduler_config = {
            'base_lr': learning_rate,
            'num_warmup_steps': config['training'].get('warmup_iters', 2000),
            'num_training_steps': config['training'].get('max_iters', 600000),
            'min_ratio': config['training'].get('min_lr_ratio', 0.1),
        }
        from models.scheduler import CosineSchedulerWithWarmup
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            **scheduler_config
        )
    elif scheduler_type == 'none':
        scheduler = None
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    init_from = config["logging"].get("init_from", "scratch")
    if init_from == "resume" and scheduler is not None:
         checkpoint = load_ckpt(config, device="cpu")
         if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
             scheduler.load_state_dict(checkpoint["scheduler"])

    return scheduler