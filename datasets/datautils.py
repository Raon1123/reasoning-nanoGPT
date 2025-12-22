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
                   split: str) -> tuple[torch.utils.data.DataLoader, dict]:
    dataset, metadata = get_dataset(config, split)
    
    dataset_config = config['dataset']
    batch_size = dataset_config.get('batch_size', 32)
    shuffle = dataset_config.get('shuffle', True) if split == 'train' else False
    num_workers = dataset_config.get('num_workers', 4)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=True)
    
    return dataloader, metadata


def get_identifiers(config: dict) -> int:
    dataset_config = config['dataset']
    data_path = dataset_config.get('data_path', './data/arc_agi/processed_data')
    identifiers_path = os.path.join(data_path, 'identifiers.json')
    
    with open(identifiers_path, 'r') as f:
        identifiers = json.load(f)
        
    num_identifiers = len(identifiers)
        
    return num_identifiers