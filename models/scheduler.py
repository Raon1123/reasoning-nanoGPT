import math

from torch.optim.lr_scheduler import LRScheduler


class NanoGPTScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_iters, lr_decay_iters, min_lr, max_lr, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch + 1
        if it < self.warmup_iters:
            return [self.max_lr * (it + 1) / (self.warmup_iters + 1) for _ in self.optimizer.param_groups]
        if it > self.lr_decay_iters:
            return [self.min_lr for _ in self.optimizer.param_groups]
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.max_lr - self.min_lr) for _ in self.optimizer.param_groups]

    def state_dict(self):
        state = super().state_dict()
        state.update({
            'warmup_iters': self.warmup_iters,
            'lr_decay_iters': self.lr_decay_iters,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr
        })
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.warmup_iters = state.get('warmup_iters', self.warmup_iters)
        self.lr_decay_iters = state.get('lr_decay_iters', self.lr_decay_iters)
        self.min_lr = state.get('min_lr', self.min_lr)
        self.max_lr = state.get('max_lr', self.max_lr)