from config import PRETRAINED_DIR
import logging
import os

import torch.nn as nn

from .src.attentive_pooler import AttentiveClassifier
from .src.utils.pretrained import load_checkpoint
from .src.vision_transformer import vit_large, vit_huge
from .src.utils.remap import VJEPA_REVERSE_MAPPING


class VJEPA(nn.Module):
    def __init__(self, pretrain_size='large') -> None:
        super(VJEPA, self).__init__()
        
        if pretrain_size == 'large':
            vit = vit_large
            backbone_path = f'{PRETRAINED_DIR}/vitl16_state.pt'
            head_path = f'{PRETRAINED_DIR}/k400-probe.pth.tar'
            input_dim = 1024
            head_dim = 400
        elif pretrain_size == 'huge':
            vit = vit_huge
            backbone_path = f'{PRETRAINED_DIR}/vith16_state.pt'
            head_path = f'{PRETRAINED_DIR}/vith-classifier_state.pt'
            input_dim = 1280
            head_dim = 400
        else:
            raise ValueError("VJEPA size must be either be 'large' or 'huge'")

        backbone_path = os.path.join(os.getcwd(), backbone_path)
        self.backbone = vit(num_frames=16) # Needs to be 16 to load_state_dict
        self.head = AttentiveClassifier(embed_dim=input_dim, num_heads=16, num_classes=head_dim)
        self.backbone = load_checkpoint(backbone_path, self.backbone)
        # self.head = load_checkpoint(head_path, self.head, is_head=True)
        self.model = nn.Sequential(self.backbone, self.head)
        self.head_without_classif = self.head.get_submodule('pooler')

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
        features = self.backbone(x)
        # out = self.head_without_classif(features)
        return features


class VJEPASwapopt(nn.Module):
    def __init__(self, pretrain_size='large') -> None:
        super(VJEPASwapopt, self).__init__()
        
        if pretrain_size == 'large':
            vit = vit_large
            backbone_path = f'/mnt/scratch/ytang/datasets/vitl16_jepa_videomix2m.pt'
            head_path = f'{PRETRAINED_DIR}/k400-probe.pth.tar'
            input_dim = 1024
            head_dim = 400
        elif pretrain_size == 'huge':
            vit = vit_huge
            backbone_path = f'{PRETRAINED_DIR}/vith16_state.pt'
            head_path = f'{PRETRAINED_DIR}/vith-classifier_state.pt'
            input_dim = 1280
            head_dim = 400
        else:
            raise ValueError("VJEPA size must be either be 'large' or 'huge'")

        self.backbone = vit(num_frames=16) # Needs to be 16 to load_state_dict
        self.head = AttentiveClassifier(embed_dim=input_dim, num_heads=16, num_classes=head_dim)
        self.backbone = load_checkpoint(backbone_path, self.backbone, remove_module=True)
        # self.head = load_checkpoint(head_path, self.head, is_head=True)
        self.model = nn.Sequential(self.backbone, self.head)
        self.head_without_classif = self.head.get_submodule('pooler')

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
        features = self.backbone(x)
        return features
