import torch
from torch import nn

from hdp_nuplan.utils.train_utils import (
    load_encoder_warm_start,
    set_encoder_trainable,
)


class TinyPlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 2)
        self.decoder = nn.Linear(2, 1)


def test_encoder_warm_start_loads_encoder_only(tmp_path):
    model = TinyPlanner()
    decoder_before = {
        key: value.clone()
        for key, value in model.decoder.state_dict().items()
    }
    checkpoint = {
        "ema_state_dict": {
            "module.encoder.weight": torch.full_like(model.encoder.weight, 2.0),
            "module.encoder.bias": torch.full_like(model.encoder.bias, 3.0),
            "module.decoder.weight": torch.full_like(model.decoder.weight, 4.0),
            "module.decoder.bias": torch.full_like(model.decoder.bias, 5.0),
        }
    }
    checkpoint_path = tmp_path / "source.pth"
    torch.save(checkpoint, checkpoint_path)

    report = load_encoder_warm_start(model, checkpoint_path)

    assert report["source_state"] == "ema_state_dict"
    assert report["loaded_tensor_count"] == 2
    assert report["target_encoder_tensor_count"] == 2
    assert report["decoder_tensor_count_loaded"] == 0
    assert report["missing_keys"] == []
    assert report["shape_mismatches"] == []
    assert torch.all(model.encoder.weight == 2.0)
    assert torch.all(model.encoder.bias == 3.0)
    for key, value in model.decoder.state_dict().items():
        assert torch.equal(value, decoder_before[key])


def test_encoder_can_be_frozen_and_unfrozen():
    model = TinyPlanner()

    set_encoder_trainable(model, False)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())

    set_encoder_trainable(model, True)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
