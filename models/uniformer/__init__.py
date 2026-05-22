import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .model import uniformer_base


class UniFormer(nn.Module):
    def __init__(self, pretrained=True) -> None:
        super(UniFormer, self).__init__()
        
        self.model = uniformer_base()

        if pretrained:
            model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_base_sthv2_32_prek600.pth")
            state_dict = torch.load(model_path, map_location='cpu')
            # Remove classifier weights
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
        return self.model(x)
