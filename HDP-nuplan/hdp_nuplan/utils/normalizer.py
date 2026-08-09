from copy import copy, deepcopy
import torch

from hdp_nuplan.utils.train_utils import openjson

class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        # HDP 只归一化自车目标；原实现还按 predicted_neighbor_num 复制邻车统计量。
        mean = [data["ego"]["mean"]]
        std = [data["ego"]["std"]]
        return cls(mean, std)
    
    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist()
        }


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
        ndt = {}
        for k, v in data.items():
            if k not in ["ego", "neighbor"]:
                ndt[k]= {"mean": torch.tensor(v["mean"], dtype=torch.float32), "std": torch.tensor(v["std"], dtype=torch.float32)}
        return cls(ndt)

    def __call__(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            try:
                # v 在 from_json/Config 中已经转换为 Tensor，直接迁移设备即可；
                # 避免 torch.tensor(Tensor) 的重复构造告警。
                mean = torch.as_tensor(v["mean"], device=data[k].device)
                std = torch.as_tensor(v["std"], device=data[k].device)
                norm_data[k] = (data[k] - mean) / std
            except Exception as e:
                raise RuntimeError(f"Error processing key '{k}': {str(e)}") from e
            norm_data[k][mask] = 0
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:  # Check if key `k` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data

    def to_dict(self):
        return {k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()} for k, v in self._normalization_dict.items()}
