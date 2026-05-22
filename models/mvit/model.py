from dataclasses import dataclass
from typing import List

from torchvision.models.video import MViT

@dataclass
class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: List[int]
    kernel_kv: List[int]
    stride_q: List[int]
    stride_kv: List[int]

def _mvit(block_setting, stochastic_depth_prob, weights, progress, **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "spatial_size", weights.meta["min_size"])
        _ovewrite_named_param(kwargs, "temporal_size", weights.meta["min_temporal_size"])
    spatial_size = kwargs.pop("spatial_size", (224, 224))
    temporal_size = kwargs.pop("temporal_size", 16)

    model = MViT(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        block_setting=block_setting,
        residual_pool=kwargs.pop("residual_pool", False),
        residual_with_cls_embed=kwargs.pop("residual_with_cls_embed", True),
        rel_pos_embed=kwargs.pop("rel_pos_embed", False),
        proj_after_attn=kwargs.pop("proj_after_attn", False),
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

def _ovewrite_named_param(kwargs, param, new_value):
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value