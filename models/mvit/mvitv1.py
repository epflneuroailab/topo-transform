from typing import Dict, List

from torch import nn
from torchvision.models.video import MViT_V1_B_Weights

from .model import MSBlockConfig, _mvit

config: Dict[str, List] = {
    "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
    "input_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
    "output_channels": [192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768, 768],
    "kernel_q": [[], [3, 3, 3], [], [3, 3, 3], [], [], [], [], [], [], [], [], [], [], [3, 3, 3], []],
    "kernel_kv": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    "stride_q": [[], [1, 2, 2], [], [1, 2, 2], [], [], [], [], [], [], [], [], [], [], [1, 2, 2], []],
    "stride_kv": [
        [1, 8, 8],
        [1, 4, 4],
        [1, 4, 4],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
    ],
}

block_setting = []
for i in range(len(config["num_heads"])):
    block_setting.append(
        MSBlockConfig(
            num_heads=config["num_heads"][i],
            input_channels=config["input_channels"][i],
            output_channels=config["output_channels"][i],
            kernel_q=config["kernel_q"][i],
            kernel_kv=config["kernel_kv"][i],
            stride_q=config["stride_q"][i],
            stride_kv=config["stride_kv"][i],
        )
    )

class MViTV1(nn.Module):
    def __init__(self, pretrained=True):
        super(MViTV1, self).__init__()

        if pretrained:
            weights = MViT_V1_B_Weights.KINETICS400_V1
            weights = MViT_V1_B_Weights.verify(weights)
            self.model = _mvit(
                block_setting=block_setting,
                residual_pool=False,
                residual_with_cls_embed=False,
                stochastic_depth_prob=0.2,
                weights=weights,
                progress=True
            )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
        x = self.model(x)
        return x

def _unsqueeze(x, target_dim, expand_dim):
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim