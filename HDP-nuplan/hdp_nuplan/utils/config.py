import json
import torch

from hdp_nuplan.utils.normalizer import StateNormalizer, ObservationNormalizer


class Config:
    
    def __init__(
            self,
            args_file,
    ):
        # HDP 的配置不再接收 guidance_fn；碰撞 guidance 是当前 Diffusion-Planner 的可选分支。
        with open(args_file, 'r') as f:
            args_dict = json.load(f)
            
        for key, value in args_dict.items():
            setattr(self, key, value)
        self.state_normalizer = StateNormalizer(self.state_normalizer['mean'], self.state_normalizer['std'])
        self.observation_normalizer = ObservationNormalizer({
            k: {
                'mean': torch.as_tensor(v['mean']),
                'std': torch.as_tensor(v['std'])
            } for k, v in self.observation_normalizer.items()
        })
