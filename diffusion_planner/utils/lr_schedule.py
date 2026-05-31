# 从 PyTorch 中导入三种学习率调度器
# SequentialLR：可以把多个 scheduler 按顺序串联起来使用
# LinearLR：线性调整学习率，常用于 warm up 阶段
# MultiplicativeLR：按给定 lambda 函数乘法调整学习率，这里用于保持学习率不变
from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiplicativeLR


# 定义一个带 warm up 的学习率调度器构造函数
# 函数名虽然叫 CosineAnnealingWarmUpRestarts ，但当前代码实际上没有使用 CosineAnnealingLR
# 当前实现逻辑是：
# 1. 前 warm_up_epoch 个 epoch 使用 LinearLR 做线性 warm up；
# 2. warm up 结束后使用 MultiplicativeLR，并令 lr_lambda 恒为 1.0，也就是保持学习率不变。
def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1):

    # 检查总 epoch 数必须大于等于 warm up epoch 数
    # 否则 warm up 阶段比总训练轮数还长，逻辑上不合理
    assert epoch >= warm_up_epoch

    # 记录 warm up 阶段的长度
    # 后面作为 SequentialLR 的切换里程碑
    T_warmup = warm_up_epoch
    
    # 定义 warm up 调度器
    # LinearLR 会从 start_factor * base_lr 开始，线性增加到 base_lr
    # start_factor=0.1 表示初始学习率是基础学习率的 10%
    # total_iters=warm_up_epoch - 1 表示线性 warm up 的迭代次数
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)

    # 定义 warm up 之后的固定学习率调度器
    # MultiplicativeLR 会在每次 step 时将学习率乘以 lr_lambda(epoch)
    # 这里 lambda epoch: 1.0 表示每次都乘以 1.0，因此学习率保持不变
    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    # 使用 SequentialLR 将 warmup_scheduler 和 fixed_scheduler 串联起来
    # 在 milestone=T_warmup 时，从 warmup_scheduler 切换到 fixed_scheduler
    scheduler = SequentialLR(optimizer, 
                             schedulers=[warmup_scheduler, fixed_scheduler], 
                             milestones=[T_warmup])
    
    # 返回组合后的学习率调度器
    return scheduler