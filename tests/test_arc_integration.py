import os
import json
import numpy as np
# import pytest
from data.arc_agi.build_arc_dataset import main as build_arc_main, DataProcessConfig
from data.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata

def test_arc_integration(tmp_path):
    # 1. Create dummy ARC data
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    
    # Create a dummy challenge file
    challenges = {
        "task1": {
            "train": [
                {"input": [[0, 1], [2, 3]], "output": [[4, 5], [6, 7]]}
            ],
            "test": [
                {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]}
            ]
        }
    }
    
    with open(input_dir / "dummy_train_challenges.json", "w") as f:
        json.dump(challenges, f)
        
    # 2. Run build_arc_dataset
    config = DataProcessConfig(
        input_file_prefix=str(input_dir / "dummy"),
        output_dir=str(output_dir),
        subsets=["train"],
        test_set_name="dummy_test",
        seed=42,
        num_aug=1,
        puzzle_identifiers_start=1
    )
    
    build_arc_main(config)
    
    # 3. Verify output files exist
    assert (output_dir / "train" / "dataset.json").exists()
    assert (output_dir / "train" / "all__inputs.npy").exists()
    
    # 4. Initialize PuzzleDataset
    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=[str(output_dir)],
        global_batch_size=2,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    dataset = PuzzleDataset(dataset_config, split="train")
    
    # 5. Iterate and verify
    iterator = iter(dataset)
    try:
        set_name, batch, batch_size = next(iterator)
        print(f"Successfully loaded batch from set: {set_name}")
        print(f"Batch keys: {batch.keys()}")
        print(f"Inputs shape: {batch['inputs'].shape}")
        
        assert "inputs" in batch
        assert "labels" in batch
        assert batch["inputs"].shape[0] == 2 # Global batch size
        
    except StopIteration:
        raise Exception("Dataset iterator is empty")

if __name__ == "__main__":
    # Manually run if executed as script
    import sys
    from pathlib import Path
    import shutil
    
    tmp_dir = Path("tmp_test_arc")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    
    try:
        test_arc_integration(tmp_dir)
        print("Test passed!")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
