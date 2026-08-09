import os
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    # 原 diffusion_planner 环境可能只安装 TensorBoard。
    # use_wandb=False 时允许日志自动退化，不阻塞监督训练和 RL 微调。
    wandb = None

class TensorBoardLogger():
    def __init__(self, run_name, notes, args, wandb_resume_id, save_path, rank=0):
        """
        project_name (str): wandb project name
        config: dict or argparser
        """              
        self.args = args
        self.writer = None
        self.id = None
        
        if rank == 0:
            if wandb is not None:
                os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
                wandb_writer = wandb.init(
                    project='Diffusion-Planner',
                    name=run_name,
                    notes=notes,
                    resume="allow",
                    id=wandb_resume_id,
                    sync_tensorboard=True,
                    dir=f'{save_path}',
                )
                wandb.config.update(args)
                self.id = wandb_writer.id
            elif args.use_wandb:
                raise RuntimeError(
                    "use_wandb=True，但当前环境未安装 wandb；"
                    "请安装 wandb 或设置 --use_wandb false"
                )
            
            self.writer = SummaryWriter(log_dir=f'{save_path}/tb')
    
    def log_metrics(self, metrics: dict, step: int):
       """
       metrics (dict):
       step (int, optional): epoch or step
       """
       if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def finish(self):
       if self.writer is not None:
            self.writer.close()
