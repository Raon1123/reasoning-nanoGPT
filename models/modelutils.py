"""
Include model and training configuration utilities.
"""
import os
import inspect
from typing import Union

import torch
from adam_atan2_pytorch import AdamAtan2

import models.nanogpt as nanogpt
import models.trm as trm
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
    elif model_type == 'trm':
        _model_config['num_identifiers'] = num_identifiers
        _model_config['vocab_size'] = vocab_size
        _model_config['batch_size'] = batch_size
        _model_config['ignore_label_id'] = ignore_label_id
        trm_config = trm.TRMConfig(**_model_config)
        model = trm.TRM(trm_config)
        if init_from == 'resume' and rank == 0:
            model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    is_compile = config['model'].get('compile', False)
    if is_compile:
        model = torch.compile(model)  # type: ignore[assignment]

    model.to(device)

    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                torch.distributed.broadcast(param.data, src=0)

    return model


def get_optimizer(config: dict,
                  model: torch.nn.Module,
                  device: torch.device,
                  world_size: int=1) -> torch.optim.Optimizer:
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
        # fix: use .parameters() not .buffers() for CastedSparseEmbedding
        puzzle_emb = getattr(model, 'puzzle_emb')
        puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            puzzle_emb.parameters(),
            lr=1e-6,
            weight_decay=0.1,
            world_size=world_size
        )
        adam_optimizer = AdamAtan2(model.parameters(), **optim_config)
        optimizer = CombinedOptimizer(puzzle_emb_optimizer, adam_optimizer)
    elif optimizer_type == 'puzzle':
        print("We are using Puzzle optimizer.")
        puzzle_config = optimizer_config.get('puzzle_config', {})
        assert 'puzzle_lr' in puzzle_config, "puzzle_lr must be specified in puzzle_config"
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        if use_fused:
            optim_config['fused'] = True

        puzzle_emb = getattr(model, 'puzzle_emb')
        puzzle_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            puzzle_emb.parameters(),
            lr=puzzle_config.get('puzzle_lr', 1e-3),
            weight_decay=puzzle_config.get('weight_decay', 0.1),
            world_size=world_size
        )
        model_optimizer = torch.optim.AdamW(
            model.parameters(),
            **optim_config
        )
        optimizer = CombinedOptimizer(puzzle_optimizer, model_optimizer)
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
            # fix B6: rank is now a proper parameter
            optimizer.load_state_dict(checkpoint['optimizer'])

    return optimizer


def get_scheduler(config: dict,
                  optimizer: Union[torch.optim.Optimizer, CombinedOptimizer]) -> Union[torch.optim.lr_scheduler.LRScheduler, CombinedScheduler, None]:
    scheduler_type = config['training']['scheduler'].get('type', 'compose').lower()

    if isinstance(optimizer, CombinedOptimizer):
        opt1 = optimizer.opt1
        opt2 = optimizer.opt2
        scheduler1 = get_scheduler(config, opt1)
        scheduler2 = get_scheduler(config, opt2)
        if scheduler1 is None and scheduler2 is None:
            return None
        return CombinedScheduler([scheduler1, scheduler2])

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
        # fix B7: read from scheduler.config, not top-level training config
        scheduler_config = config['training']['scheduler'].get('config', {})
        scheduler_init_config = {
            'base_lr': learning_rate,
            'num_warmup_steps': scheduler_config.get('warmup_iters', 2000),
            'num_training_steps': config['training'].get('max_iters', 600000),
            'min_ratio': scheduler_config.get('min_ratio', 0.1),
        }
        from models.scheduler import CosineSchedulerWithWarmup
        scheduler = CosineSchedulerWithWarmup(optimizer, **scheduler_init_config)
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
