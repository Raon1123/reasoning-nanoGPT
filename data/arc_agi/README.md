Run the following command to prepare the ARC-AGI dataset for puzzle reasoning at root of the project:
```bash
uv run python -m data.arc_agi.prepare \
    --input-file-prefix ./data/arc_agi/kaggle/input/arc-agi \
    --output-dir ./data/arc_agi/processed_data \
    --subsets concept training evaluation \
    --test-set-name evaluation
```