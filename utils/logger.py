import os

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
        
        log_dir = logging_config.get('logdir', 'logs')
        
        if self.wandb_log:
            wandb.init(project=project,
                       name=run_name,
                       config=config)
        
        self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
        
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
        