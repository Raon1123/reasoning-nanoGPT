# ARC-AGI datasets

## How to run

```bash
uv run build-arc-dataset
```

Or with custom arguments:

```bash
uv run python -m data.arc_agi.prepare \
  --input-file-prefix kaggle/input \
  --output-dir data/arc-aug-1000 \
  --subsets concept training evaluation \
  --test-set-name evaluation
```

## Brief code overview

### prepare.py


### common.py

This `common.py` contains utility functions for augmentation and meta-data for puzzle datasets.