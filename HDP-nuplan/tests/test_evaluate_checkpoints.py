from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from evaluate_checkpoints import checkpoint_sort_key, rank_records  # noqa: E402


def test_rank_records_orders_validation_loss_without_mutating_input():
    records = [
        {"checkpoint": "epoch-1", "metrics": {"loss": 0.4}},
        {"checkpoint": "epoch-2", "metrics": {"loss": 0.2}},
        {"checkpoint": "epoch-3", "metrics": {"loss": 0.3}},
    ]

    ranked = rank_records(records)

    assert [item["checkpoint"] for item in ranked] == ["epoch-2", "epoch-3", "epoch-1"]
    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert all("rank" not in item for item in records)


def test_checkpoint_sort_key_uses_numeric_epoch_order():
    paths = [
        Path("model_epoch_10_trainloss_0.1.pth"),
        Path("model_epoch_2_trainloss_0.2.pth"),
        Path("model_epoch_1_trainloss_0.3.pth"),
    ]

    assert [path.name for path in sorted(paths, key=checkpoint_sort_key)] == [
        "model_epoch_1_trainloss_0.3.pth",
        "model_epoch_2_trainloss_0.2.pth",
        "model_epoch_10_trainloss_0.1.pth",
    ]
