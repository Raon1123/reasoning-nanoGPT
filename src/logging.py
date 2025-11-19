import os

from torch.utils.tensorboard import SummaryWriter
# conditional wandb import
try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False
    
    
class Logger:
    def __init__(self, 
                 log_dir: str, 
                 use_wandb: bool = False, wandb_project: str = "nanoGPT", wandb_run_name: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        self.use_wandb = use_wandb and _wandb_available
        if self.use_wandb:
            wandb.init(project=wandb_project, name=wandb_run_name, dir=log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
    
    def log_histogram(self, tag: str, values, step: int):
        self.writer.add_histogram(tag, values, step)
        if self.use_wandb:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def close(self):
        self.writer.close()
        if self.use_wandb:
            wandb.finish()