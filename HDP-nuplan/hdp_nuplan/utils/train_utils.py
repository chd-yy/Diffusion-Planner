import torch
import random
import numpy as np
from mmengine import fileio
import io
import os
import json


def load_encoder_warm_start(model, checkpoint_path):
    """只加载同名且同形状的 encoder tensor，并返回完整审计报告。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        source_name = "ema_state_dict"
        source_state = checkpoint[source_name]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        source_name = "model"
        source_state = checkpoint[source_name]
    elif isinstance(checkpoint, dict):
        source_name = "root_state_dict"
        source_state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    source_state = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in source_state.items()
    }
    target_state = model.state_dict()
    loaded = {}
    missing = []
    shape_mismatch = []

    for key, target_value in target_state.items():
        if not key.startswith("encoder."):
            continue
        if key not in source_state:
            missing.append(key)
            continue
        source_value = source_state[key]
        if source_value.shape != target_value.shape:
            shape_mismatch.append(
                {
                    "key": key,
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue
        loaded[key] = source_value

    source_encoder_keys = {
        key for key in source_state if key.startswith("encoder.")
    }
    unexpected = sorted(source_encoder_keys - set(loaded) - set(missing))

    merged_state = dict(target_state)
    merged_state.update(loaded)
    model.load_state_dict(merged_state, strict=True)

    return {
        "checkpoint_path": str(checkpoint_path),
        "source_state": source_name,
        "loaded_tensor_count": len(loaded),
        "loaded_parameter_count": int(sum(value.numel() for value in loaded.values())),
        "target_encoder_tensor_count": sum(
            key.startswith("encoder.") for key in target_state
        ),
        "missing_keys": sorted(missing),
        "shape_mismatches": shape_mismatch,
        "unexpected_source_encoder_keys": unexpected,
        "decoder_tensor_count_loaded": sum(
            key.startswith("decoder.") for key in loaded
        ),
    }


def set_encoder_trainable(model, trainable):
    """切换 HDP encoder 的梯度，用于 warm-start 后分阶段解冻。"""
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(trainable)

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict

def opendata(path):
    
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff)

    return npz_data

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]


    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss

def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema):
    """
    save the model to path
    """
    save_model = {'epoch': epoch + 1, 
                  'model': model.state_dict(), 
                  'ema_state_dict': ema.state_dict(),
                  'optimizer': optimizer.state_dict(), 
                  'schedule': scheduler.state_dict(), 
                  'loss': train_loss,
                  'wandb_id': wandb_id}

    with io.BytesIO() as f:
        torch.save(save_model, f)
        fileio.put(f.getvalue(), f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth')
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")

def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    load ckpt from path
    """
    path = os.path.join(path, 'latest.pth')
    ckpt = fileio.get(path)
    with io.BytesIO(ckpt) as f:
        ckpt = torch.load(f)

    # load model
    try:
        model.load_state_dict(ckpt['model'])
    except:
        model.load_state_dict(ckpt)                   
    print("Model load done")
    
    # load optimizer
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")
            
    # load schedule
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except:
        print("no schedule found,")
    
    # load step
    try:
        init_epoch = ckpt['epoch']
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict(ckpt['ema_state_dict'])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')

    return model, optimizer, scheduler, init_epoch, wandb_id, ema

