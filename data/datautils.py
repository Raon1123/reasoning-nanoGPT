
from torch.utils.data import DataLoader

# datasets...
from data.puzzledatasets import PuzzleDataset, PuzzleDatasetConfig


def get_puzzle_dataset(config: dict,
                       split: str,
                       rank: int,
                       world_size: int,
                       **kwargs) -> PuzzleDataset:
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(**config),
        split=split,
    )
    return dataset


def get_puzzle_dataloader(config: dict,
                          **kwargs) -> DataLoader:
    pass