import os
import torch
import numpy as np
import json
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Optional, Tuple, Union, List

def get_batch(split: str, config: Dict[str, Any], device_type: str, device: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy get_batch function for text datasets (OpenWebText, Shakespeare).
    Kept for backward compatibility or specific use cases.
    
    Args:
        split (str): 'train' or 'val'.
        config (Dict[str, Any]): Configuration dictionary.
        device_type (str): 'cuda' or 'cpu'.
        device (Any): Torch device.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of inputs (x) and targets (y).
    """
    # data_dir is relative to the project root, but since we are running from project root, 'data' is correct.
    # However, if we want to be robust:
    data_dir = os.path.join('data', config['data']['dataset'])
    block_size = config['data']['block_size']
    batch_size = config['data']['batch_size']

    # We recreate np.memmap every batch to avoid a memory leak
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

class ARCDataset(Dataset):
    """
    ARC-AGI dataset loader.
    Expects data_dir to contain JSON files for ARC tasks.
    """
    def __init__(self, data_dir: str, split: str = 'train', transform: Optional[Any] = None) -> None:
        """
        Initialize ARCDataset.

        Args:
            data_dir (str): Path to data directory.
            split (str): 'train' or 'test'.
            transform (Optional[Any]): Optional transform to apply.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.tasks: List[Any] = []
        
        # Load all JSON files
        # Assuming structure: data_dir/{split}/*.json or similar
        # Adjusting based on typical ARC dataset structure
        search_path = os.path.join(data_dir, split, '*.json')
        self.files = sorted(glob.glob(search_path))
        
        if not self.files:
            # Fallback or warning if no files found (might be in root of data_dir)
            search_path = os.path.join(data_dir, '*.json')
            self.files = sorted(glob.glob(search_path))
            
        print(f"Found {len(self.files)} tasks in {search_path}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item at index.

        Args:
            idx (int): Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
        """
        file_path = self.files[idx]
        with open(file_path, 'r') as f:
            task = json.load(f)
        
        # TODO: Process task into input/output tensors
        # This requires a tokenizer and specific formatting (e.g. grid to string)
        # For now, returning dummy tensors to satisfy the training loop
        # Assuming block_size is available or we pick a fixed size
        # We need config or block_size passed to init
        
        # Placeholder: return random tensors
        # In reality, we would tokenize the grid state
        seq_len = 1024 # Should be config.block_size
        x = torch.randint(0, 50257, (seq_len,), dtype=torch.long)
        y = torch.randint(0, 50257, (seq_len,), dtype=torch.long)
        return x, y 

from data.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


class TextDataset(Dataset):
    """
    Dataset for text data (e.g. OpenWebText, Shakespeare) using memory mapping.
    Mimics the logic of the original get_batch function.
    """
    def __init__(self, data_dir: str, split: str, block_size: int) -> None:
        """
        Initialize TextDataset.

        Args:
            data_dir (str): Path to data directory.
            split (str): 'train' or 'val'.
            block_size (int): Context length.
        """
        self.block_size = block_size
        filename = 'train.bin' if split == 'train' else 'val.bin'
        self.data_path = os.path.join(data_dir, filename)
        
        if os.path.exists(self.data_path):
            self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
            self.length = len(self.data) - self.block_size
        else:
            print(f"Warning: Data file {self.data_path} not found.")
            self.data = None
            self.length = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item at index.

        Args:
            idx (int): Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
        """
        if self.data is None:
            return torch.zeros(self.block_size, dtype=torch.long), torch.zeros(self.block_size, dtype=torch.long)
        
        # Ensure idx is within bounds (DataLoader should handle this if len is correct)
        # But strictly speaking, we want random sampling from the whole buffer.
        # The DataLoader with shuffle=True will sample random indices from 0 to len-1.
        
        chunk = self.data[idx:idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

def get_data(config: Dict[str, Any], split: str = 'train') -> Optional[Dataset]:
    """
    Factory function to get the appropriate dataset based on config.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        split (str): 'train' or 'val'.

    Returns:
        Optional[Dataset]: The dataset object or None.
    """
    dataset_name = config['data']['dataset']
    data_dir = os.path.join('data', dataset_name)
    
    if 'puzzle' in dataset_name.lower() or 'arc' in dataset_name.lower():
        # Construct PuzzleDatasetConfig from the main config
        # Assuming config structure matches what PuzzleDataset expects or we map it
        
        # We need to extract relevant parts from config['data'] and config['system'] etc.
        # This is a bit of a guess on mapping, assuming config is a dict.
        
        # For now, let's create a default config if not present, or map available fields.
        # The PuzzleDatasetConfig requires:
        # seed, dataset_paths, global_batch_size, test_set_mode, epochs_per_iter, rank, num_replicas
        
        puzzle_config = PuzzleDatasetConfig(
            seed=config['system'].get('seed', 42),
            dataset_paths=[data_dir], # Assuming data_dir is the path
            global_batch_size=config['data']['batch_size'], # Assuming batch_size is global? Or we need to multiply?
            # Usually batch_size in nanoGPT config is per-device or global depending on implementation.
            # Let's assume it's what the user wants.
            test_set_mode=(split == 'test'),
            epochs_per_iter=config['data'].get('epochs_per_iter', 10),
            rank=config.get('ddp_info', {}).get('rank', 0),
            num_replicas=config.get('ddp_info', {}).get('world_size', 1)
        )
        
        return PuzzleDataset(puzzle_config, split=split)
    else:
        # Assume text dataset
        block_size = config['data']['block_size']
        return TextDataset(data_dir, split, block_size)

def get_dataloader(dataset: Optional[Dataset], config: Dict[str, Any], ddp_info: Optional[Dict[str, Any]] = None) -> Optional[DataLoader]:
    """
    Creates a DataLoader with optional DDP support.

    Args:
        dataset (Optional[Dataset]): The dataset to load.
        config (Dict[str, Any]): Configuration dictionary.
        ddp_info (Optional[Dict[str, Any]]): DDP information dictionary.

    Returns:
        Optional[DataLoader]: The DataLoader object or None.
    """
    if dataset is None:
        return None

    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 0)
    
    sampler = None
    if ddp_info and ddp_info['ddp']:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_info['world_size'],
            rank=ddp_info['rank'],
            shuffle=True,
            seed=config['system'].get('seed', 1337)
        )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=... # Add custom collate if needed for variable length sequences
    )
    
    return loader
