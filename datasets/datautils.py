
import torch

from datasets.puzzle_dataset import NaiveDataset

def get_dataset(config: dict,
                split: str) -> torch.utils.data.Dataset:
    dataset_config = config['dataset']
    
    data_path = dataset_config.get('data_path', './data/puzzles/')
    
    

