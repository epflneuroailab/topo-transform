from torchvision import transforms

from .clip import CLIPVision
from .llcnn import LLCNNVision
from .mvit.mvitv1 import MViTV1
from .tdann import TDANN
from .toponets import TopoNetsVision
from .uniformer import UniFormer
from .videomae import VideoMAEVision
from .vjepa import VJEPA
from .vjepa import VJEPASwapopt


def build_vit_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img / 255.0),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_clip_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img / 255.0),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def build_llcnn_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img / 255.0),
        ]
    )


vit_transform = build_vit_transform()
clip_transform = build_clip_transform()
llcnn_transform = build_llcnn_transform()


__all__ = [
    "MViTV1",
    "CLIPVision",
    "LLCNNVision",
    "TDANN",
    "TopoNetsVision",
    "UniFormer",
    "VideoMAEVision",
    "VJEPA",
    "VJEPASwapopt",
    "build_clip_transform",
    "build_llcnn_transform",
    "build_vit_transform",
    "clip_transform",
    "llcnn_transform",
    "vit_transform",
]
