import os
import sys
from data import ARCDataset

def test_arc_dataset():
    # Create a dummy task file for testing
    os.makedirs('data/arc_agi/train', exist_ok=True)
    with open('data/arc_agi/train/dummy.json', 'w') as f:
        f.write('{"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[0]], "output": [[1]]}]}')

    dataset = ARCDataset('data/arc_agi', split='train')
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        task = dataset[0]
        print("Sample task keys:", task.keys())
        assert 'train' in task
        assert 'test' in task
        print("ARC Dataset test passed!")
    else:
        print("ARC Dataset test failed: No tasks found.")

if __name__ == "__main__":
    test_arc_dataset()
