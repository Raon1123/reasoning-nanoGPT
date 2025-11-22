import os
import random
import numpy as np
import torch
import time
from contextlib import nullcontext
from typing import Any, Optional

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_ctx(device_type: str) -> Any:
    """
    Get the context manager for autocast based on device type.

    Args:
        device_type (str): The device type ('cpu' or 'cuda').

    Returns:
        Any: The context manager (nullcontext or torch.amp.autocast).
    """
    # This needs to be passed from config or inferred, for now simple implementation
    # In real usage, we'd pass dtype from config
    if device_type == 'cpu':
        return nullcontext()
    elif device_type == 'cuda':
        # Assuming bfloat16 is desired for CUDA, or it should be passed from config
        # The original comment suggests bfloat16 is the default
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        # Fallback for other device types, or raise an error
        return nullcontext() # Or handle other device types as needed
