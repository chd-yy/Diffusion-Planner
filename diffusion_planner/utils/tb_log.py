# 导入 os 模块
# 这里主要用于设置环境变量，例如 WANDB_MODE
import os

# 从 PyTorch 的 TensorBoard 工具中导入 SummaryWriter
# SummaryWriter 用于将训练指标写入 TensorBoard 日志文件
from torch.utils.tensorboard import SummaryWriter

# 导入 wandb
# wandb 用于实验记录、指标可视化、配置保存和实验恢复
import wandb


# 定义 TensorBoardLogger 类
# 这个类主要用于统一管理 TensorBoard 和 wandb 的日志记录
class TensorBoardLogger():

    # 初始化日志记录器
    # run_name：当前实验名称
    # notes：当前实验备注
    # args：训练参数配置，通常来自 argparse
    # wandb_resume_id：wandb 的 run id，用于恢复已有实验
    # save_path：日志和 wandb 文件保存路径
    # rank：分布式训练中的进程编号，通常只让 rank=0 的主进程记录日志
    def __init__(self, run_name, notes, args, wandb_resume_id, save_path, rank=0):
        """
        project_name (str): wandb project name
        config: dict or argparser
        """              

        # 保存训练参数配置
        self.args = args

        # 初始化 TensorBoard writer
        # 默认设为 None，只有 rank=0 时才真正创建
        self.writer = None

        # 初始化 wandb run id
        # 后续如果成功初始化 wandb，会将真实 id 保存到 self.id
        self.id = None
        
        # 只在主进程 rank=0 中初始化 wandb 和 TensorBoard
        # 这样可以避免分布式训练时多个进程重复写日志
        if rank == 0:

            # 根据 args.use_wandb 决定 wandb 工作模式
            # 如果 args.use_wandb=True，则使用 online 模式，将日志同步到 wandb 云端
            # 如果 args.use_wandb=False，则使用 offline 模式，只在本地记录
            os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"

            # 初始化 wandb 实验
            # project：wandb 项目名称
            # name：当前 run 的名称
            # notes：实验备注
            # resume="allow"：允许根据 id 恢复已有 run，如果 id 不存在则创建新 run
            # id=wandb_resume_id：指定要恢复或继续记录的 wandb run id
            # sync_tensorboard=True：自动同步 TensorBoard 日志到 wandb
            # dir=f'{save_path}'：wandb 本地文件保存目录
            wandb_writer = wandb.init(project='Diffusion-Planner', 
                name=run_name, 
                notes=notes,
                resume="allow",
                id = wandb_resume_id,
                sync_tensorboard=True,
                dir=f'{save_path}')

            # 将 args 中的训练配置写入 wandb config
            # 这样可以在 wandb 页面中查看当前实验的超参数设置
            wandb.config.update(args)

            # 保存当前 wandb run 的 id
            # 后续可以在保存 checkpoint 时一并保存，用于断点恢复同一个 wandb run
            self.id = wandb_writer.id
            
            # 创建 TensorBoard SummaryWriter
            # 日志文件会保存在 save_path/tb 目录下
            self.writer = SummaryWriter(log_dir=f'{save_path}/tb')
    
    # 记录训练或验证指标
    # metrics 是一个字典，例如 {"loss": 0.123, "lr": 1e-4}
    # step 表示当前训练步数或 epoch
    def log_metrics(self, metrics: dict, step: int):
       """
       metrics (dict):
       step (int, optional): epoch or step
       """

       # 只有 self.writer 不为 None 时才记录日志
       # 如果当前不是 rank=0 进程，则 self.writer 为 None，不会写日志
       if self.writer is not None:

            # 遍历所有指标
            for key, value in metrics.items():

                # 将标量指标写入 TensorBoard
                # key 是指标名称，例如 train/loss
                # value 是指标数值
                # step 是横轴步数
                self.writer.add_scalar(key, value, step)

    # 结束日志记录
    def finish(self):

       # 如果 TensorBoard writer 存在，则关闭它
       # 关闭后会刷新缓存，确保日志文件完整写入磁盘
       if self.writer is not None:
            self.writer.close()