# 导入 PyTorch，用于模型参数保存、加载、随机种子设置等
import torch

# 导入 Python 内置 random 模块，用于控制 Python 层面的随机性
import random

# 导入 numpy，用于数组计算、随机种子设置、loss 求均值等
import numpy as np

# 从 mmengine 导入 fileio
# fileio 提供统一的文件读写接口，可以兼容本地文件、远程存储等不同后端
from mmengine import fileio

# 导入 io 模块
# 这里主要使用 io.BytesIO，在内存中读写二进制数据，避免必须先落盘
import io

# 导入 os 模块，用于路径拼接等文件路径操作
import os

# 导入 json 模块，用于读取 json 格式配置或数据列表
import json


# 读取 json 文件，并解析成 Python 字典或列表
def openjson(path):

       # 使用 mmengine.fileio 读取文本内容
       # fileio.get_text(path) 会返回 path 对应文件中的字符串内容
       value  = fileio.get_text(path)

       # 将 json 字符串反序列化为 Python 对象
       # 如果 json 文件最外层是 {}，则得到 dict
       # 如果 json 文件最外层是 []，则得到 list
       dict = json.loads(value)

       # 返回解析后的 Python 对象
       return dict


# 读取 npz 数据文件
# 支持通过 mmengine.fileio 从本地或远程路径读取
def opendata(path):
    
    # 读取 path 对应文件的原始二进制内容
    # 对于 .npz 文件，这里得到的是完整文件的 bytes
    npz_bytes = fileio.get(path)

    # 将二进制 bytes 包装成 BytesIO 对象
    # np.load 可以像读取文件一样读取这个内存缓冲区
    buff = io.BytesIO(npz_bytes)

    # 使用 numpy 加载 npz 数据
    # npz_data 类似一个字典，可以通过 key 访问其中保存的数组
    npz_data = np.load(buff)

    # 返回加载后的 npz 数据对象
    return npz_data


# 设置随机种子，尽可能保证实验可复现
def set_seed(CUR_SEED):

    # 设置 Python 内置 random 模块的随机种子
    random.seed(CUR_SEED)

    # 设置 numpy 的随机种子
    np.random.seed(CUR_SEED)

    # 设置 PyTorch CPU 随机种子
    torch.manual_seed(CUR_SEED)

    # 设置 cuDNN 使用确定性算法
    # 这样可以减少同一代码多次运行结果不一致的问题
    torch.backends.cudnn.deterministic = True

    # 关闭 cuDNN benchmark 自动优化
    # benchmark=True 时，cuDNN 会根据输入尺寸自动选择最快算法，但可能引入非确定性
    torch.backends.cudnn.benchmark = False


# 计算一个 epoch 中各类 loss 的平均值
# epoch_loss 通常是一个列表，列表中每个元素是一个 dict
# 例如：
# epoch_loss = [
#     {"loss": tensor(1.2), "cls_loss": tensor(0.3)},
#     {"loss": tensor(1.0), "cls_loss": tensor(0.2)}
# ]
def get_epoch_mean_loss(epoch_loss):

    # 用于收集每一种 loss 在当前 epoch 内的所有数值
    epoch_mean_loss = {}

    # 遍历每个 step 或 batch 记录的 loss 字典
    for current_loss in epoch_loss:

        # 遍历当前 batch 中的每个 loss 项
        for key, value in current_loss.items():

            # 如果该 loss 名称之前已经出现过
            if key in epoch_mean_loss:

                # 将当前 loss 值加入对应列表
                # 如果 value 本身是 int 或 float，则直接使用
                # 如果 value 是 torch.Tensor，则调用 item() 转成 Python 标量
                epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())

            # 如果该 loss 名称第一次出现
            else:

                # 新建一个列表保存该 loss 的数值
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]


    # 遍历每一种 loss，把列表中的多个 batch loss 求平均
    for key, values in epoch_mean_loss.items():

        # 将 values 转成 numpy 数组后求均值
        epoch_mean_loss[key] = np.mean(np.array(values))

    # 返回每种 loss 的 epoch 平均值
    return epoch_mean_loss


# 保存模型 checkpoint
# checkpoint 中不仅保存模型参数，也保存优化器、学习率调度器、EMA、训练轮数、loss 和 wandb id
def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema):
    """
    save the model to path
    """

    # 构造需要保存的 checkpoint 字典
    save_model = {'epoch': epoch + 1, 
                  'model': model.state_dict(), 
                  'ema_state_dict': ema.state_dict(),
                  'optimizer': optimizer.state_dict(), 
                  'schedule': scheduler.state_dict(), 
                  'loss': train_loss,
                  'wandb_id': wandb_id}

    # 创建一个内存中的二进制缓冲区
    # torch.save 会先把 checkpoint 写入这个 BytesIO，而不是直接写入磁盘
    with io.BytesIO() as f:

        # 将 checkpoint 序列化保存到内存缓冲区 f 中
        torch.save(save_model, f)

        # 将缓冲区中的二进制内容写入指定路径
        # 文件名包含 epoch 和 train_loss，便于区分不同训练轮次的模型
        fileio.put(f.getvalue(), f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth')

        # 同时再保存一份 latest.pth
        # latest.pth 始终表示最新 checkpoint，方便断点续训时直接加载
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")


# 从 latest.pth 恢复模型、优化器、学习率调度器、EMA 等状态
def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    load ckpt from path
    """

    # 拼接 checkpoint 路径
    # 默认从 path/latest.pth 加载
    path = os.path.join(path, 'latest.pth')

    # 使用 fileio 读取 checkpoint 的二进制内容
    ckpt = fileio.get(path)

    # 将二进制内容包装成 BytesIO 对象，供 torch.load 读取
    with io.BytesIO(ckpt) as f:

        # 从内存缓冲区中反序列化 checkpoint
        ckpt = torch.load(f)

    # load model
    # 加载模型参数
    try:

        # 常规情况：ckpt 是一个字典，其中 'model' 保存模型参数
        model.load_state_dict(ckpt['model'])

    except:

        # 兼容另一种情况：ckpt 本身就是模型 state_dict
        model.load_state_dict(ckpt)                   

    # 打印模型加载完成提示
    print("Model load done")
    
    # load optimizer
    # 加载优化器状态
    try:

        # 从 checkpoint 中恢复 optimizer 的 state_dict
        optimizer.load_state_dict(ckpt['optimizer'])

        # 打印优化器加载完成提示
        print("Optimizer load done")

    except:

        # 如果 checkpoint 中没有 optimizer，说明可能只是预训练权重，不包含训练状态
        print("no pretrained optimizer found")
            
    # load schedule
    # 加载学习率调度器状态
    try:

        # 从 checkpoint 中恢复 scheduler 的 state_dict
        scheduler.load_state_dict(ckpt['schedule'])

        # 打印调度器加载完成提示
        print("Schedule load done")

    except:

        # 如果 checkpoint 中没有 schedule，则跳过
        print("no schedule found,")
    
    # load step
    # 加载训练轮数
    try:

        # 读取 checkpoint 中保存的 epoch
        # 这通常作为恢复训练时的起始 epoch
        init_epoch = ckpt['epoch']

        # 打印 epoch 加载完成提示
        print("Step load done")

    except:

        # 如果 checkpoint 中没有 epoch，则从 0 开始
        init_epoch = 0

    # Load wandb id
    # 加载 wandb 实验 id，用于恢复同一个 wandb run
    try:

        # 读取 checkpoint 中保存的 wandb_id
        wandb_id = ckpt['wandb_id']

        # 打印 wandb id 加载完成提示
        print("wandb id load done")

    except:

        # 如果没有 wandb_id，则设为 None
        wandb_id = None

    # 加载 EMA 模型参数
    try:

        # 将 checkpoint 中保存的 EMA 参数加载到 ema.ema 中
        ema.ema.load_state_dict(ckpt['ema_state_dict'])

        # 将 EMA 模型设置为 eval 模式
        ema.ema.eval()

        # 冻结 EMA 模型参数
        # EMA 通常只用于评估或推理，不直接参与梯度更新
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        # 打印 EMA 加载完成提示
        print("ema load done")

    except:

        # 如果 checkpoint 中没有 EMA 参数，则跳过
        print('no ema shadow found')

    # 返回恢复后的对象和训练状态
    return model, optimizer, scheduler, init_epoch, wandb_id, ema