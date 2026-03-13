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

    # NOTE: LRScheduler.__init__ calls step() once internally, setting last_epoch=0.
    # Every subsequent step() increments last_epoch by 1.
    # NanoGPTScheduler.get_lr uses it = last_epoch + 1, so:
    #   - After __init__: last_epoch=0, it=1
    #   - After 1st manual step(): last_epoch=1, it=2
    scheduler = NanoGPTScheduler(optimizer, warmup_iters, lr_decay_iters, min_lr, max_lr)
    # At init: last_epoch=0, it=1, warmup (1 < 2): lr = max_lr * (1+1)/(2+1) = max_lr*2/3
    expected_lr_init = max_lr * (1 + 1) / (warmup_iters + 1)
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr_init) < 1e-6, \
        f"After init: Expected {expected_lr_init}, got {optimizer.param_groups[0]['lr']}"

    # Step 1: last_epoch=1, it=2; 2 < 2 = False → cosine decay, ratio=0 → lr = max_lr
    scheduler.step()
    expected_lr = min_lr + 1.0 * (max_lr - min_lr)  # coeff=1.0 at ratio=0
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, \
        f"After step 1: Expected {expected_lr:.6f}, got {optimizer.param_groups[0]['lr']:.6f}"

    # Step 2: last_epoch=2, it=3; cosine decay, ratio=(3-2)/(10-2)=1/8
    scheduler.step()
    decay_ratio = (3 - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    expected_lr = min_lr + coeff * (max_lr - min_lr)
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, \
        f"After step 2 (decay): Expected {expected_lr:.6f}, got {optimizer.param_groups[0]['lr']:.6f}"

    # Advance to last decay step: last_epoch=9, it=10
    for _ in range(7):  # steps 3..9
        scheduler.step()
    decay_ratio = (10 - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    expected_lr = min_lr + coeff * (max_lr - min_lr)
    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6, \
        f"At end of decay: Expected {expected_lr:.6f}, got {optimizer.param_groups[0]['lr']:.6f}"

    # Step beyond lr_decay_iters: last_epoch=10, it=11 > 10 → min_lr
    scheduler.step()
    assert abs(optimizer.param_groups[0]['lr'] - min_lr) < 1e-6, \
        f"Beyond decay: Expected {min_lr}, got {optimizer.param_groups[0]['lr']}"

    print("All tests passed!")

if __name__ == "__main__":
    test_scheduler()
