
# reasoning-nanoGPT

In this repository we build baseline for reasoning tasks with Transformers.
Base repository is [nanoGPT](https://github.com/karpathy/nanoGPT).

## install

In this project, we manage our library dependencies with [uv](https://astral.sh/uv/). It is a lightweight environment and dependency management tool that works on all major platforms. To install uv and sync the environment, run:

```
# install uv for environment management
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Enjoy!

## Prepare datasets

### ARC-AGI datasets

At `data/arc_agi`, we expect following structure:

```
# ARC-AGI datasets

```

### Sudoku datasets


## todos

- [ ] add training and evaluation scripts for reasoning tasks
- [ ] upgrade nanochat model to support distributed training
- [ ] add code tests... (not prioritized)
- [ ]


## acknowledgements

Baseline codes from following repositories

- [NanoGPT](https://github.com/karpathy/nanoGPT)
- Hierarchical Reasoning Model ([HRM](https://github.com/sapientinc/HRM)) and analysis HRM from ARC teams [HRM Analysis](https://github.com/arcprize/hierarchical-reasoning-model-analysis)
- Tiny Recursive Model ([TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels))
