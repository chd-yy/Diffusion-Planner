from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiplicativeLR

def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1):
    assert epoch >= warm_up_epoch
    if warm_up_epoch <= 1:
        # 没有可执行的线性 warm-up 区间时保持调用方设置的基础学习率。
        # LinearLR(total_iters=0) 会意外把学习率永久缩放为 start_factor。
        return MultiplicativeLR(optimizer, lr_lambda=lambda _: 1.0)

    T_warmup = warm_up_epoch
    
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    scheduler = SequentialLR(optimizer, 
                             schedulers=[warmup_scheduler, fixed_scheduler], 
                             milestones=[T_warmup])
    
    return scheduler
