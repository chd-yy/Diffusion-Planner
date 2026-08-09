import pytest
import torch

from hdp_nuplan.utils.lr_schedule import CosineAnnealingWarmUpRestarts


def test_single_epoch_warmup_keeps_configured_learning_rate():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=5e-4)
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        epoch=2,
        warm_up_epoch=1,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
