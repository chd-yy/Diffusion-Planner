# 导入 os 模块
# 主要用于读取和设置环境变量，例如 RANK、WORLD_SIZE、LOCAL_RANK、MASTER_PORT 等
import os

# 导入 PyTorch
# 这里主要用于获取 GPU 数量、设置当前 GPU 设备等
import torch

# 导入 torch.distributed 模块
# 用于分布式训练中的进程通信、同步、all_reduce 等操作
import torch.distributed as dist

# 从 torch.distributed 中导入 init_process_group
# 用于初始化分布式进程组
from torch.distributed import init_process_group

# 导入 subprocess
# 这里主要用于在 SLURM 环境下执行 shell 命令，获取主节点地址
import subprocess


# 通用 DDP 初始化函数
# 兼容三种情况：
# 1. args.ddp=False：不使用 DDP，直接单卡训练；
# 2. torchrun / torch.distributed.launch 启动：环境变量中存在 RANK、WORLD_SIZE；
# 3. SLURM 集群启动：环境变量中存在 SLURM_PROCID。
def ddp_setup_universal(verbose=False, args=None):

       # 如果用户显式关闭 DDP，则不初始化分布式训练
       # 返回 rank=0、gpu=0、world_size=1，表示单进程单卡训练
       if args.ddp == False:
              print(f"do not use ddp, train on GPU 0")
              return 0, 0, 1
       
       # 情况 1：通过 torchrun 或 torch.distributed.launch 启动
       # 这类启动方式通常会自动设置 RANK、WORLD_SIZE、LOCAL_RANK 环境变量
       if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:

              # 当前进程的全局 rank
              # rank 用于标识当前进程在所有分布式进程中的编号
              rank = int(os.environ["RANK"])

              # 总进程数
              # world_size 通常等于总 GPU 数或总训练进程数
              world_size = int(os.environ['WORLD_SIZE'])

              # 当前进程使用的本地 GPU 编号
              # LOCAL_RANK 通常表示当前机器上的 GPU id
              gpu = int(os.environ['LOCAL_RANK'])

              # 设置主节点通信端口
              # 如果 args 中有 port，则使用 args.port；否则默认使用 29529
              os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))

              # 设置主节点地址
              # 这里写成 localhost，通常适合单机多卡训练
              os.environ["MASTER_ADDR"] = "localhost"

       # 情况 2：通过 SLURM 集群任务启动
       # SLURM 会设置 SLURM_PROCID、SLURM_NTASKS、SLURM_NODELIST 等环境变量
       elif 'SLURM_PROCID' in os.environ:

              # 当前进程在 SLURM 任务中的全局编号
              rank = int(os.environ['SLURM_PROCID'])

              # 根据当前 rank 和本机 GPU 数量确定当前进程使用哪张 GPU
              gpu = rank % torch.cuda.device_count()

              # SLURM 总任务数，对应分布式训练的 world_size
              world_size = int(os.environ['SLURM_NTASKS'])

              # 获取 SLURM 分配的节点列表
              node_list = os.environ['SLURM_NODELIST']

              # 获取当前机器可见 GPU 数量
              num_gpus = torch.cuda.device_count()

              # 通过 scontrol 命令解析 SLURM 节点列表，取第一个节点作为主节点地址
              addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')

              # 设置主节点通信端口
              os.environ['MASTER_PORT'] = str(args.port)

              # 设置主节点地址
              os.environ['MASTER_ADDR'] = addr

       # 情况 3：既不是 torchrun，也不是 SLURM
       # 则认为不使用 DDP
       else:
              print("Not using DDP mode")
              return 0, 0, 1

       # 将 world_size 写回环境变量
       # 后续 init_process_group 使用 env:// 初始化时会依赖这些环境变量
       os.environ['WORLD_SIZE'] = str(world_size)

       # 将当前本地 GPU 编号写回环境变量
       os.environ['LOCAL_RANK'] = str(gpu)

       # 将当前全局 rank 写回环境变量
       os.environ['RANK'] = str(rank)              

       # 设置当前进程使用的 GPU
       # 每个 DDP 进程通常绑定一张 GPU
       torch.cuda.set_device(gpu)

       # 设置分布式通信后端
       # NCCL 是 NVIDIA GPU 上最常用、性能较高的通信后端
       dist_backend = 'nccl'

       # 使用环境变量方式初始化分布式训练
       # env:// 会从 MASTER_ADDR、MASTER_PORT、RANK、WORLD_SIZE 中读取配置
       dist_url = "env://"

       # 打印当前分布式初始化信息
       print('| distributed init (rank {}): {}, gpu {}'.format(rank, dist_url, gpu), flush=True)

       # 初始化分布式进程组
       # backend：通信后端
       # world_size：总进程数
       # rank：当前进程编号
       init_process_group(backend=dist_backend, world_size=world_size, rank=rank)

       # 设置同步屏障
       # 确保所有进程都完成初始化后再继续执行
       torch.distributed.barrier()

       # 如果 verbose=True，则设置非主进程禁止普通 print
       # 这样可以避免多进程训练时日志重复打印
       if verbose:
              setup_for_distributed(rank == 0)

       # 返回当前进程的 rank、GPU 编号和总进程数
       return rank, gpu, world_size


# 设置分布式打印行为
# 非主进程默认不打印，避免多卡训练时日志刷屏
def setup_for_distributed(is_master):
       """
       This function disables printing when not in master process
       """

       # 导入 Python 内置模块 builtins
       # 用于重写全局 print 函数
       import builtins as __builtin__

       # 保存原始 print 函数
       builtin_print = __builtin__.print

       # 重新定义 print 函数
       def print(*args, **kwargs):

              # 从 kwargs 中取出 force 参数
              # force=True 时，即使不是主进程也强制打印
              force = kwargs.pop('force', False)

              # 只有主进程或 force=True 时才真正调用原始 print
              if is_master or force:
                     builtin_print(*args, **kwargs)

       # 用新的 print 函数替换 Python 内置 print
       __builtin__.print = print


# 获取当前分布式训练的总进程数
def get_world_size():

       # 如果分布式不可用或未初始化，则认为是单进程训练，返回 1
       if not is_dist_avail_and_initialized():
              return 1

       # 返回当前分布式进程组中的总进程数
       return dist.get_world_size()


# 获取当前进程的全局 rank
def get_rank():

       # 如果分布式不可用或未初始化，则认为当前是主进程，rank=0
       if not is_dist_avail_and_initialized():
              return 0

       # 返回当前进程在分布式进程组中的 rank
       return dist.get_rank()


# 获取真正的模型对象
# 在 DDP 包装后，原始模型会被放在 model.module 中
def get_model(model, use_ddp):

       # 如果使用 DDP，则返回 DDP 包装内部的原始模型
       if use_ddp:
              return model.module

       # 如果没有使用 DDP，则直接返回模型本身
       else:
              return model


# 判断当前环境是否支持并且已经初始化分布式训练
def is_dist_avail_and_initialized():

       # 如果 torch.distributed 不可用，则返回 False
       if not dist.is_available():
              return False

       # 如果 torch.distributed 尚未初始化，则返回 False
       if not dist.is_initialized():
              return False

       # 分布式可用且已初始化
       return True




# 对各个进程上的 loss 进行求和并取平均
# 常用于 DDP 训练中统计所有 GPU 上的平均 loss
def reduce_and_average_losses(loss_dict, device):

       # 同步所有进程
       # 确保各进程都计算完 loss 后再进行 all_reduce
       torch.distributed.barrier()

       # 获取当前分布式训练总进程数
       world_size = dist.get_world_size()

       # 遍历 loss 字典中的每一个 loss 项
       for key in loss_dict.keys():

              # 将当前 loss 转成 tensor，并移动到指定 device
              # loss_dict[key].item() 将标量 tensor 转成 Python 数值
              loss_tensor = torch.tensor([loss_dict[key].item()]).to(device)

              # 对所有进程中的 loss_tensor 做求和
              # all_reduce 后，每个进程都会得到所有进程 loss 的总和
              dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

              # 除以 world_size，得到所有进程的平均 loss
              # 再写回 loss_dict
              loss_dict[key] = loss_tensor.item() / world_size

       # 返回同步平均后的 loss 字典
       return loss_dict