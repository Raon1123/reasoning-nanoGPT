from contextlib import nullcontext

import torch

from datasets.datautils import get_dataloader, get_identifiers
from models.modelutils import get_model, get_optimizer, get_scheduler
from models.optimizer import CombinedOptimizer
from utils.const import IGNORE_LABEL_ID
from utils.epochs import eval_epoch
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
    # fix B3: use configured dtype instead of float32 (which made autocast a no-op)
    _dtype_str = config.get('model', {}).get('dtype', 'float16')
    _dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    _autocast_dtype = _dtype_map.get(_dtype_str, torch.float16)
    autocast_ctx = torch.autocast(device_type=device_type, dtype=_autocast_dtype) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1 * 8)

    # logger
    logger = Logger(config) if master_process else None
    always_save = config['logging'].get('always_save', False)
    best_val_loss = float('inf') if master_process else None
    eval_only = config['logging'].get('eval_only', False) if master_process else False

    # data loader
    train_loader, train_metadata = get_dataloader(config, split='train', ddp_local_rank=ddp_local_rank, world_size=ddp_world_size)

    # test/val loaders are only needed on master process for evaluation
    if not master_process:
        val_loader = None
        test_loader = None
    else:
        test_batch_size = config['training'].get('test_batch_size', 32)
        # val_loader intentionally uses the train split to monitor training-set metrics
        val_loader, _ = get_dataloader(config, split='train', ddp_local_rank=ddp_local_rank, world_size=ddp_world_size, batch_size=test_batch_size)
        test_loader, _ = get_dataloader(config, split='test', ddp_local_rank=ddp_local_rank, world_size=ddp_world_size, batch_size=test_batch_size)
    num_identifiers = get_identifiers(config)

    max_iters = config['training'].get('max_iters', 100000)
    eval_interval = config['logging'].get('eval_interval', 1000)

    ignore_label_id = IGNORE_LABEL_ID
    meta_vocab_size = train_metadata.get('vocab_size', 12)
    batch_size = config['training'].get('batch_size', 32)

    model = get_model(config,
                      device.type,
                      num_identifiers=num_identifiers,
                      vocab_size=meta_vocab_size,
                      batch_size=batch_size,
                      ignore_label_id=ignore_label_id,
                      world_size=ddp_world_size,
                      rank=ddp_rank)

    optimizer = get_optimizer(config, model, device, world_size=ddp_world_size)
    scheduler = get_scheduler(config, optimizer)

    pbar = range(max_iters)
    if master_process:
        from tqdm import tqdm
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True)

    epoch = 0
    # fix B4: initialize train_iter before the loop to avoid NameError on first iteration
    train_iter = iter(train_loader)

    for iter_num in pbar:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if ddp:
                train_loader.sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

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

        # evaluation here
        if iter_num % eval_interval == 0 and master_process:
            assert val_loader is not None and test_loader is not None
            assert logger is not None
            eval_metrics = evaluation(config, model, val_loader, test_loader, device)
            eval_metrics['iter_num'] = iter_num
            logger.log_metrics(eval_metrics, step=iter_num)

            if eval_metrics['test/loss'] < best_val_loss or always_save:
                best_val_loss = eval_metrics['test/loss']
                if iter_num > 0:
                    if ddp:
                        synchronize()
                    logger.log_ckpt(model,
                                    optimizer,
                                    scheduler,
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
                model.require_backward_grads_sync = (micro_step == gradient_accumulation_steps - 1)
            with autocast_ctx:
                logits, loss = model(X, puzzle_ids, Y, test_mode=False)
                loss = loss / gradient_accumulation_steps
            loss.backward()

        # all reduce grads
        if ddp and gradient_accumulation_steps > 1:
            for param in model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad)

        # gradient clipping
        grad_clip = config['training'].get('optimizer', {}).get('grad_clip', None)
        if grad_clip is not None and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if isinstance(optimizer, CombinedOptimizer):
            optimizer.opt1.step()
            optimizer.opt2.step()
            optimizer.opt1.zero_grad(set_to_none=True)
            optimizer.opt2.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        if master_process:
            if iter_num % eval_interval == 0:
                # fix B5: guard get_last_lr() call
                if scheduler is not None and hasattr(scheduler, 'get_last_lr'):
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.0
                pbar.set_description(f"iter {iter_num}, loss {loss.item():.4f}, lr {current_lr:.2e}")
                logger.log_metrics({'train/loss': loss.item(),
                                    'train/lr': current_lr,
                                    'train/iter': iter_num},
                                   step=iter_num)

    if ddp:
        torch.distributed.destroy_process_group()


def evaluation(config: dict,
               model: torch.nn.Module,
               trn_dataloader: torch.utils.data.DataLoader,
               tst_dataloader: torch.utils.data.DataLoader,
               device: torch.device) -> dict:
    out = {}
    model.eval()
    for split in ['train', 'test']:
        dataloader = trn_dataloader if split == 'train' else tst_dataloader
        eval_ret = eval_epoch(config, model, dataloader, device)
        for key, value in eval_ret.items():
            out[f"{split}/{key}"] = value
    model.train()
    return out


if __name__ == "__main__":
    args = get_args()
    config = load_yaml(args.config)
    main(config)
