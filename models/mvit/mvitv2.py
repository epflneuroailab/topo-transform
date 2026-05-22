from typing import override, Dict, List

from torch import nn
from torchvision.models.video import MViT_V2_S_Weights

from models.build import MODEL_REGISTRY
from models.mvit.model import MSBlockConfig, _mvit
from models.visionmodel import VisionModel

config: Dict[str, List] = {
    "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
    "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
    "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
    "kernel_q": [
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
    "stride_q": [
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
    ],
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

@MODEL_REGISTRY.register()
class MViTV2(VisionModel):
    def __init__(self, cfg, ds_classes):
        super(MViTV2, self).__init__(ds_classes)

        if cfg.MODEL.PRETRAINED:
            weights = MViT_V2_S_Weights.KINETICS400_V1
            weights = MViT_V2_S_Weights.verify(weights)
            self.model = _mvit(
                block_setting=block_setting,
                residual_pool=True,
                residual_with_cls_embed=False,
                rel_pos_embed=True,
                proj_after_attn=True,
                stochastic_depth_prob=cfg.MODEL.MVIT.STOCH_DEPTH_PROB,
                weights=weights,
                progress=True
        )
        else:
            self.model = _mvit(
                spatial_size=(cfg.DATA.WIDTH, cfg.DATA.HEIGHT),
                temporal_size=cfg.DATA.FPS * cfg.DATA.DURATION // 1000,
                block_setting=block_setting,
                residual_pool=True,
                residual_with_cls_embed=False,
                rel_pos_embed=True,
                proj_after_attn=True,
                stochastic_depth_prob=cfg.MODEL.MVIT.STOCH_DEPTH_PROB,
                progress=True,
                weights=None
            )

        # TODO get size of last layer dynamically
        self.heads = self.create_classifiers(cfg, 768, ds_classes)

    @override
    def extract_features(self, x):
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.model.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.model.pos_encoding.temporal_size,) + self.model.pos_encoding.spatial_size
        for block in self.model.blocks:
            x, thw = block(x, thw)
        x = self.model.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x
    
    @override
    def create_classifiers(self, cfg, in_features, ds_classes):
        return nn.ModuleDict({
            name: nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features=in_features, out_features=ds_classes[name], bias=True)
            )
            for name in cfg.DATASETS.TRAINING.NAMES
        })
    
    @override
    def get_classifiers(self):
        return self.heads

def _unsqueeze(x, target_dim, expand_dim):
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim