import torch
import re
import torch.nn as nn
from torchvision.models import resnet18


class TDANN(nn.Module):
    def __init__(self) -> None:
        super(TDANN, self).__init__()
        self.model = resnet18()

    def load_pretrained_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_params = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]

        adjusted_model_params = {}
        pattern = re.compile(r"(\d+)_(\d+)")

        for key, value in model_params.items():
            new_key = pattern.sub(r'\1.\2', key)
            adjusted_model_params[new_key] = value

        adjusted_model_params = {k.replace("base_model.", ""): v for k, v in adjusted_model_params.items()}
        adjusted_model_params = {k.replace("_feature_blocks.", ""): v for k, v in adjusted_model_params.items()}

        msg = self.model.load_state_dict(adjusted_model_params, strict=False)
        print(("Pretrained weights found at {} and loaded with msg: {}".format(ckpt_path, msg)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
