"""
Test code for datasets.py
"""
import pytest
from data.datasets import PuzzleDatasetMetadata


def test_default_dataset_paths():
    """Test that the default dataset_paths is ['data/arc_agi']"""
    config = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        vocab_size=12,
        seq_len=900,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=1.0,
        sets=["all"]
    )