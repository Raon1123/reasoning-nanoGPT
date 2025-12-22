import os

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    def __init__(self,
                 config: dict) -> None:
        
        logging_config = config['logging']
        
        self.wandb_log = logging_config.get('wandb_log', False) and WANDB_AVAILABLE
        
        project = logging_config.get('wandb_project', 'reasoning-nanogpt')
        run_name = logging_config.get('wandb_run_name', 'default_run')
        
        self.run_dir = get_run_dir(config)
        os.makedirs(self.run_dir, exist_ok=True)
        
        if self.wandb_log:
            wandb.init(project=project,
                       name=run_name,
                       config=config,
                       dir=self.run_dir)
        
        self.tb_writer = SummaryWriter(log_dir=self.run_dir)
        
    def log_metrics(self,
                    metrics: dict,
                    step: int) -> None:
        for key, value in metrics.items():
            self.tb_writer.add_scalar(key, value, step)
            if self.wandb_log:
                wandb.log({key: value}, step=step)
                
    def close(self) -> None:
        self.tb_writer.close()
        if self.wandb_log:
            wandb.finish()
            
    def log_image(self,
                  tag: str,
                  img_tensor,
                  step: int) -> None:
        self.tb_writer.add_image(tag, img_tensor, step)
        if self.wandb_log:
            wandb.log({tag: wandb.Image(img_tensor)}, step=step)    
            
    def log_ckpt(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 model_args: dict,
                 iter_num: int,
                 best_val_loss: float,
                 config: dict,) -> None:
        
        ckpt_path = get_ckpt_path(config)
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config
        }
        torch.save(checkpoint, ckpt_path)


def get_run_dir(config: dict) -> str:
    logging_config = config['logging']
    
    run_name = logging_config.get('wandb_run_name', 'default_run')
    log_dir = logging_config.get('logdir', 'logs')
    run_dir = os.path.join(log_dir, run_name)
    
    return run_dir
   
        
def get_ckpt_path(config: dict) -> str:
    ckpt_path = os.path.join(
        get_run_dir(config),
        'ckpt.pt'
    )
    return ckpt_path   
        
def load_ckpt(config: dict,
              device: str) -> dict:
    ckpt_path = get_ckpt_path(config)
    checkpoint = torch.load(ckpt_path, map_location=device)
    return checkpoint