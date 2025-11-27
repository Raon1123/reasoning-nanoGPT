import os
from contextlib import nullcontext

import torch

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
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    
    # logger
    logger = Logger(config) if master_process else None
    
    model = get_model(config, device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])


if __name__ == "__main__":
    args = get_args()
    config = load_yaml(args.config)
    main(config)
