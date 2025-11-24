import math
import sys
import torch

sys.path.append('.')
from models.scheduler import NanoGPTScheduler

def test_scheduler():
    # Create a dummy optimizer
    optimizer = torch.optim.SGD([torch.tensor([1.0], requires_grad=True)], lr=0.1)

    # Scheduler parameters
    warmup_iters = 2
    lr_decay_iters = 10
    min_lr = 0.01
    max_lr = 0.1

    scheduler = NanoGPTScheduler(optimizer, warmup_iters, lr_decay_iters, min_lr, max_lr)
    scheduler.last_epoch = -1  # Start from -1, like PyTorch default

    # Test warmup phase
    # At iteration 0 (after first step)
    scheduler.step()
    expected_lr = max_lr * (0 + 1) / (warmup_iters + 1)
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, f"Expected {expected_lr}, got {optimizer.param_groups[0]['lr']}"

    # At iteration 1
    scheduler.step()
    expected_lr = max_lr * (1 + 1) / (warmup_iters + 1)
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, f"Expected {expected_lr}, got {optimizer.param_groups[0]['lr']}"

    # At iteration 2 (end of warmup)
    scheduler.step()
    expected_lr = max_lr * (2 + 1) / (warmup_iters + 1)
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, f"Expected {expected_lr}, got {optimizer.param_groups[0]['lr']}"

    # Test decay phase
    # At iteration 3
    scheduler.step()
    decay_ratio = (3 - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    expected_lr = min_lr + coeff * (max_lr - min_lr)
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, f"Expected {expected_lr}, got {optimizer.param_groups[0]['lr']}"

    # At iteration 10 (end of decay)
    for _ in range(7):  # steps 4 to 10
        scheduler.step()
    decay_ratio = (10 - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    expected_lr = min_lr + coeff * (max_lr - min_lr)
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, f"Expected {expected_lr}, got {optimizer.param_groups[0]['lr']}"

    # At iteration 11 (beyond decay_iters)
    scheduler.step()
    expected_lr = min_lr
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, f"Expected {expected_lr}, got {optimizer.param_groups[0]['lr']}"

    print("All tests passed!")

if __name__ == "__main__":
    test_scheduler()