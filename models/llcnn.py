import importlib
import sys
import types
from pathlib import Path

import torch
from torch import nn


LLCNN_ROOT = Path("/mnt/scratch/ytang/LLCNN")
LLCNN_DEFAULT_CKPT = (
    LLCNN_ROOT
    / "cache/gaussian_0.23"
    / "cln_resnet18contopo_gaussian_1_0.0gaussian_0.23_continuous_prog_t"
    / "resnet18contopo_100.pt"
)


def _ensure_llcnn_package():
    """Expose LLCNN's model files under a private package name."""
    package_name = "_llcnn_external"
    models_name = f"{package_name}.models"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(LLCNN_ROOT)]
        sys.modules[package_name] = package
    if models_name not in sys.modules:
        models_package = types.ModuleType(models_name)
        models_package.__path__ = [str(LLCNN_ROOT / "models")]
        sys.modules[models_name] = models_package
    return package_name


class LLCNNVision(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        package_name = _ensure_llcnn_package()
        module = importlib.import_module(f"{package_name}.models.resnet_imagenet_continuoustopo")
        self.checkpoint_path = Path(checkpoint_path or LLCNN_DEFAULT_CKPT)
        self.model = module.ResNet18(
            1000,
            pool_type="gaussian",
            max_num_pools=1,
            noise_std=0.0,
            kap_kernelsize=0.23,
            continuous=True,
            local_conv=False,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.replace("module.", ""): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

    def forward(self, x):
        return self.model(x)
