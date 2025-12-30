import os
import json

import torch

from datasets.puzzle_dataset import NaiveDataset
from utils.const import IGNORE_LABEL_ID

def get_dataset(config: dict,
                split: str) -> tuple[torch.utils.data.Dataset, dict]:
    dataset_config = config['dataset']
    
    data_path = dataset_config.get('data_path', './data/arc_agi/processed_data')
    
    # split must be one of 'train', 'test'
    assert split in ['train', 'test'], "split must be one of 'train' or 'test'"
    
    data_root = os.path.join(data_path, split)
    metadata = json.load(open(os.path.join(data_root, 'dataset.json')))
    dataset = NaiveDataset(data_root=data_root,
                           metadata=metadata,
                           ignore_label_id=IGNORE_LABEL_ID)
    
    return dataset, metadata


def get_dataloader(config: dict,
                   split: str,
                   world_size: int=1,
                   ddp_local_rank: int=-1) -> tuple[torch.utils.data.DataLoader, dict]:
    dataset, metadata = get_dataset(config, split)
    
    dataloader_config = config['dataset']['config']
    batch_size = config['training'].get('batch_size', 32)
    shuffle = dataloader_config.get('shuffle', True) 
    num_workers = dataloader_config.get('num_workers', 4)
    pin_memory = dataloader_config.get('pin_memory', False)
    
    print(f"Creating {split} dataloader with batch size {batch_size}, "
          f"shuffle={shuffle}, num_workers={num_workers}, pin_memory={pin_memory}")
    
    if ddp_local_rank >= 0:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=ddp_local_rank,
            shuffle=shuffle
        )
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 sampler=sampler,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory)
    
    return dataloader, metadata


def get_identifiers(config: dict) -> int:
    dataset_config = config['dataset']
    data_path = dataset_config.get('data_path', './data/arc_agi/processed_data')
    identifiers_path = os.path.join(data_path, 'identifiers.json')
    
    with open(identifiers_path, 'r') as f:
        identifiers = json.load(f)
        
    num_identifiers = len(identifiers)
        
    return num_identifiers