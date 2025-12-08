import os
from contextlib import nullcontext

import torch

from datasets.datautils import get_dataloader
from models.modelutils import get_model, get_optimizer, get_scheduler
from utils.logger import Logger
from utils.toolkit import (
    compute_init, compute_cleanup,
    load_yaml
)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('--config', '-c', type=str, default='config/config_default.yaml',
                        help='Path to a config file (YAML) that specifies training parameters')
    args = parser.parse_args()
    return args
 

def main(config: dict):
    device_type = config.get('device_type', 'cuda')
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type=device_type)
    
    master_process = (not ddp) or (ddp and ddp_rank == 0)
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    scaler = torch.GradScaler(device=device.type) 
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 5 * 8)
    
    # logger
    logger = Logger(config) if master_process else None
    always_save = config['logging'].get('always_save', False)
    best_val_loss = float('inf') if master_process else None
    eval_only = config['logging'].get('eval_only', False) if master_process else False
    
    model = get_model(config, device.type)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
        
    # data loader
    train_loader, train_metadata = get_dataloader(config, split='train')
    test_loader, test_metadata = get_dataloader(config, split='test')
    
    max_iters = config['training'].get('max_iters', 100000)
    eval_interval = config['training'].get('eval_interval', 1000)
    ckpt_interval = config['training'].get('ckpt_interval', 1000)
    
    optimizer = get_optimizer(config, model, device)
    scheduler = get_scheduler(config, optimizer)
    
    pbar = range(max_iters)
    if master_process:
        from tqdm import tqdm
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True)
        
    for iter_num in pbar:
        model.train()
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # evaluation here
        if iter_num % eval_interval == 0 and master_process:
            eval_metrics = eval_epoch(config,
                                      model,
                                      train_loader,
                                      test_loader,
                                      device,
                                      ignore_padding=True)
            eval_metrics['iter_num'] = iter_num
            logger.log_metrics(eval_metrics, step=iter_num)
            
            if eval_metrics['test/loss'] < best_val_loss or always_save:
                best_val_loss = eval_metrics['test/loss']
                if iter_num > 0:
                    logger.log_ckpt(model.module if ddp else model,
                                    optimizer,
                                    model_args=config['model']['config'],
                                    iter_num=iter_num,
                                    best_val_loss=best_val_loss,
                                    config=config)
                if iter_num == 0 and eval_only:
                    print("Eval only mode, exiting after first eval.")
                    compute_cleanup()
                    return
                
        # train here
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with autocast_ctx:
                logits, loss = model(*batch, test_mode=False)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            
        # clip
        grad_clip = config['training'].get('grad_clip', 1.0)
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        synchronize()
        
        if master_process:
            if iter_num % 100 == 0:
                mem = get_max_memory() / (1024 ** 3) if device_type == "cuda" else 0
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_description(f"iter {iter_num}, loss {loss.item():.4f}, lr {current_lr:.2e}, mem {mem:.2f} GB")
                logger.log_metrics({'train/loss': loss.item(),
                                    'train/lr': current_lr,
                                    'train/mem': mem},
                                   step=iter_num)
                
    if ddp:
        torch.distributed.destroy_process_group()
                
        
    

def eval_epoch(config: dict,
               model: torch.nn.Module,
               trn_dataloader: torch.utils.data.DataLoader,
               tst_dataloader: torch.utils.data.DataLoader,
               device: torch.device,
               ignore_padding: bool) -> dict:
    out = {}
    model.eval()
    for split in ['train', 'test']:
        dataloader = trn_dataloader if split == 'train' else tst_dataloader

        eval_iters = config['training'].get('eval_iters', 200)
        losses = torch.zeros(eval_iters)
        valid_puzzle_counts = 0
        pixel_accuracies = torch.zeros(eval_iters)
        exact_corrects = 0
        
        for k in range(eval_iters):
            try:
                batch = next(eval_iter)
            except:
                eval_iter = iter(dataloader)
                batch = next(eval_iter)
                
            X, Y, puzzle_ids = batch
            # apply pin memory and device, non_blocking
            if device.type != 'cpu':
                X = X.pin_memory().to(device, non_blocking=True)
                Y = Y.pin_memory().to(device, non_blocking=True)
                puzzle_ids = puzzle_ids.pin_memory().to(device, non_blocking=True)
            else:
                X = X.to(device)
                Y = Y.to(device)
                puzzle_ids = puzzle_ids.to(device)
                
            with torch.inference_mode():
                logits, loss = model(X, puzzle_ids, Y, test_mode=True)
                
                if ignore_padding:
                    # if ignore padding, padding is 0 in Y
                    mask = (Y != 0)
                else:
                    mask = torch.ones_like(Y, dtype=torch.bool)
                
                preds = torch.argmax(logits, dim=-1)
                correct = (preds == Y) & mask
                pixel_accuracy = correct.sum().item() / mask.sum().item()
                exact_accuracy = correct.sum(-1) == mask.sum(-1)
            
            losses[k] = loss.item()
            pixel_accuracies[k] = pixel_accuracy
            exact_corrects += exact_accuracy.sum().item()
            valid_puzzle_counts += Y.size(0)
            
        out[f'{split}/loss'] = losses.mean().item()
        out[f'{split}/pixel_accuracy'] = pixel_accuracies.mean().item()
        out[f'{split}/exact_accuracy'] = exact_corrects / valid_puzzle_counts
        
    model.train()
        
    return out


if __name__ == "__main__":
    args = get_args()
    config = load_yaml(args.config)
    main(config)
