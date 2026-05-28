import importlib
import sys
from pathlib import Path

import torch
from torch import nn


TOPONETS_ROOT = Path("/mnt/scratch/ytang/toponets")
TOPONETS_DEFAULT_CKPT = (
    Path("/mnt/scratch/ytang/topotransform/cache/checkpoints/toponets")
    / "resnet18_tau_10.0.pt"
)


def _ensure_toponets_path():
    root = str(TOPONETS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class TopoNetsVision(nn.Module):
    """TopoNets canonical vision model: ResNet18 trained with TopoLoss at tau=10."""

    def __init__(self, tau=10.0, checkpoint_path=None):
        super().__init__()
        _ensure_toponets_path()
        module = importlib.import_module("toponets")
        self.tau = float(tau)
        self.checkpoint_path = Path(checkpoint_path or TOPONETS_DEFAULT_CKPT)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = module.resnet18(tau=self.tau, checkpoint_path=str(self.checkpoint_path))

    def forward(self, x):
        return self.model(x.contiguous(memory_format=torch.channels_last))
