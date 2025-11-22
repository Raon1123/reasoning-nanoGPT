import os
import time
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Logger class for handling WandB and TensorBoard logging.
    """
    def __init__(self, config: Dict[str, Any], master_process: bool = True) -> None:
        """
        Initialize the logger.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            master_process (bool): Whether this is the master process (only master logs).
        """
        self.config = config
        self.master_process = master_process
        self.wandb_run = None
        self.tb_writer = None
        
        if not self.master_process:
            return

        self.wandb_log: bool = config['logging']['wandb_log']
        self.wandb_project: str = config['logging']['wandb_project']
        self.wandb_run_name: str = config['logging']['wandb_run_name']
        self.out_dir: str = config['system']['out_dir']

        if self.wandb_log:
            import wandb
            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=config)
            self.wandb_run = wandb
        
        # Always setup tensorboard if master process
        os.makedirs(self.out_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=self.out_dir)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics to available loggers.

        Args:
            metrics (Dict[str, float]): Dictionary of scalar metrics to log.
            step (int): Current training step/iteration.
        """
        if not self.master_process:
            return

        # Log to WandB
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

    def close(self) -> None:
        """
        Close all loggers.
        """
        if not self.master_process:
            return
            
        if self.wandb_run:
            self.wandb_run.finish()
        
        if self.tb_writer:
            self.tb_writer.close()
