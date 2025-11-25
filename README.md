
# reasoning-nanoGPT

---
We can find TODO from the comments `TODO` in the code. I should miss some points... Don't forget to check them.
TODOS:
- [ ]: Make a dataloader!!!! Don't use binfile for loading data. (FOR BETTER DATA LOADING)
- [ ]: In `train.py`, it hardcode about the configs. We must manage our configs with yaml or etc. (DON'T HARDCODE THE CONFIG FOR REPRODUCIBILITY)
- [ ]: Use `float16` instead of `bfloat16` for better compatibility with GPUs.
- [ ]: Add more optimizer such as Adam_Aten2 etc.
- []: Make batch with `X,y` manner `X=[input puzzles, puzzle_identifiers]`

---

- At `datasets/puzzle_dataset.py`, removed `set_name` and `global_effective_batch_size` from yielded values in `__iter__` method of `PuzzleDataset` class. What is the `puzzle_identifiers` for then?

## install

This project library is managed by [uv](https://docs.astral.sh/uv/). To install it, first make sure you have `uv` installed (see the link for instructions). Then simply run:

```sh
uv sync
```

<details>
<summary> How to install uv? </summary>
`uv` is a modern package manager for Python projects. To install it, you can use the following command in your terminal:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
</details>

### Prepare datasets

see subdirectory `data/` for dataset preparation scripts.

## quick start



```bash
OMP_NUM_THREADS=4 uv run torchrun --nproc_per_node=4 train.py --config config/config_default.yaml
```


