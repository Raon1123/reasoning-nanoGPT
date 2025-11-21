
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# datasets...
from data.puzzledatasets import PuzzleDataset, PuzzleDatasetConfig


def get_puzzle_dataset(config: dict,
                       split: str,
                       seed: int,
                       rank: int,
                       world_size: int,
                       **kwargs) -> PuzzleDataset:
    
    puzzle_config = config.get('DATASET', {})
    
    puzzle_config['seed'] = seed
    
    puzzle_config['global_batch_size'] = config['TRAIN'].get('batch_size', 32) 
    puzzle_config['test_set_mode'] = (split != 'train')
    
    eval_interval = config['LOGGING'].get('eval_interval', 2000)
    puzzle_config['epochs_per_iter'] = eval_interval if split == 'train' else 1 
    
    puzzle_config['rank'] = rank
    num_replicas = world_size if world_size > 1 else 1
    puzzle_config['num_replicas'] = num_replicas
    
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(**puzzle_config),
        split=split,
    )
    return dataset


def get_puzzle_dataloader(config: dict,
                          split: str,
                          seed: int,
                          rank: int,
                          world_size: int,
                          **kwargs) -> DataLoader:
    dataset = get_puzzle_dataset(config, 
                                 split, 
                                 seed,
                                 rank, 
                                 world_size, 
                                 **kwargs)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    local_batch_size = config['TRAIN'].get('batch_size', 32) // world_size if world_size > 1 else config['TRAIN'].get('batch_size', 32)
    
    dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=config.get('NUM_WORKERS', 1),
        pin_memory=False,
    )
    return dataloader