import os
import torch
from torch.distributed import init_process_group

def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get('RANK', -1)) != -1


def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def setup_ddp(config, device):
    ddp = is_ddp()
    if ddp:
        ddp_config = config.get('DDP', {})
        backend = ddp_config.get('backend', 'nccl')
        
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        
        gradient_accumulation_steps = ddp_config.get('gradient_accumulation_steps', 1)
        assert gradient_accumulation_steps % ddp_world_size == 0, \
            "Gradient accumulation steps must be divisible by world size."
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        gradient_accumulation_steps = config.get('DDP', {}).get('gradient_accumulation_steps', 1)
        
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size
    
    if master_process:
        print(f"Using device: {device}")
        print(f"DDP enabled: {ddp}, World Size: {ddp_world_size}")
        print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"Tokens per iteration: {tokens_per_iter}")
    
    return ddp, device, master_process, seed_offset, gradient_accumulation_steps, tokens_per_iter, ddp_world_size
    

def fix_seeds(seed: int,
              offset: int = 0) -> None:
    torch.manual_seed(seed + offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True