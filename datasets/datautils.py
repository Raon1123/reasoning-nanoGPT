
from datasets.puzzle_dataset import PuzzleDatasetConfig, PuzzleDataset

def get_dataset(config: dict, 
                split: str):
    dataset_config = config['dataset']['config']
    
    puzzle_config = {
        'seed': dataset_config.get('seed', 1337),
        'dataset_path': dataset_config.get('data_root', 'data/arc_agi_aug1000'),
        'global_batch_size': config['training']['global_batch_size'],
        'epochs_per_iter': dataset_config.get('epochs_per_iter'),
        'rank': dataset_config.get('rank', 0),
        'num_replicas': dataset_config.get('num_replicas', 1),
        'test_set_mode': (split == 'test')
    }
    puzzle_config = PuzzleDatasetConfig(**puzzle_config)
    
    dataset = PuzzleDataset(config=puzzle_config, split=split)
    return dataset

