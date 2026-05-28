import config

import torch
from topo import TopoTransformedCLIP
from topo import TopoTransformedLLCNN
from topo import TopoTransformedTDANN
from topo import TopoTransformedTopoNets
from topo import TopoTransformedVideoMAE
from topo import TopoTransformedVJEPA
from topo import SOMTopoVJEPA
from .autocorr import validate_autocorr
from .floc import validate_floc
from .floc import validate_floc_human
from .invertible import validate_invertibility


def _get_seed(ckpt_name):
    import re
    match = re.search(r"sd(\d+)", ckpt_name)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r"seed(\d+)", ckpt_name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Could not extract seed from checkpoint name: {ckpt_name}")


def _is_tdann_checkpoint(checkpoint_name):
    return "tdann." in checkpoint_name


def _is_swapopt_checkpoint(checkpoint_name):
    return checkpoint_name.startswith("swapopt")


def _is_som_checkpoint(checkpoint_name):
    return checkpoint_name.startswith("som_")


def _is_clip_checkpoint(checkpoint_name):
    return "clip_" in checkpoint_name


def _is_videomae_checkpoint(checkpoint_name):
    return "videomae_" in checkpoint_name


def _is_llcnn_checkpoint(checkpoint_name):
    return checkpoint_name.startswith("llcnn.")


def _is_toponets_checkpoint(checkpoint_name):
    return checkpoint_name.startswith("toponets.")


def _resolve_tissue_config(checkpoint_name):
    tissue_config = "small" if "_small" in checkpoint_name else "vtc"
    import re
    match = re.search(r"_rf([0-9p]+)", checkpoint_name)
    rf_overlap = float(match.group(1).replace("p", ".")) if match else None
    inf_neighborhood = "_neighbInf" in checkpoint_name and tissue_config != "small"
    return tissue_config, rf_overlap, inf_neighborhood


def _resolve_vjepa_checkpoint(checkpoint_name):
    if checkpoint_name.startswith("unoptimized."):
        return config.CACHE_DIR / "checkpoints" / checkpoint_name.replace("unoptimized.", ""), True
    return config.CACHE_DIR / "checkpoints" / checkpoint_name, False


def _resolve_layer_indices(checkpoint_name):
    import re
    match = re.search(r"(?:vjepa|clip|videomae)_([0-9_]+)_single", checkpoint_name)
    if match:
        return [int(layer_index) for layer_index in match.group(1).split("_")]
    return [14, 18, 22] if "14_18_22" in checkpoint_name else [18]


def _load_tdann_model(checkpoint_name):
    seed = int(checkpoint_name.split("_seed")[-1].split(".")[0])
    checkpoint_path = config.CACHE_DIR / "checkpoints" / checkpoint_name.replace("tdann.", "tdann/")
    model = TopoTransformedTDANN(seed=seed)
    model.name = checkpoint_name
    model.model.load_pretrained_weights(checkpoint_path)
    print(f"Loaded TopoTransformedTDANN model from {checkpoint_path}.")
    return model, None


def _load_swapopt_model(checkpoint_name):
    single_sheet = checkpoint_name.startswith("swapopt_single_sheet")
    layer_indices = [14, 18, 22] if single_sheet else [18]
    model = TopoTransformedVJEPA(
        layer_indices=layer_indices,
        swapopt=True,
        inf_neighborhood=False,
        single_sheet=single_sheet,
        seed=_get_seed(checkpoint_name),
    )
    model.name = checkpoint_name
    return model, None


def _load_som_model(checkpoint_name, device):
    checkpoint_path = config.CACHE_DIR / "checkpoints" / checkpoint_name
    model = SOMTopoVJEPA(layer_indices=_resolve_layer_indices(checkpoint_name), seed=_get_seed(checkpoint_name))
    model.name = checkpoint_name or checkpoint_path.stem
    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = model.load_state_dict(checkpoint["som_model_state_dict"], strict=False)
    epoch = checkpoint.get("epoch")
    print(f"Loaded SOMTopoVJEPA model from {checkpoint_path} (epoch {epoch}).")
    print(msg)
    return model, epoch


def _load_clip_model(checkpoint_name, device):
    checkpoint_path, no_transform = _resolve_vjepa_checkpoint(checkpoint_name)
    tissue_config, rf_overlap, inf_neighborhood = _resolve_tissue_config(checkpoint_name)
    model = TopoTransformedCLIP(
        layer_indices=_resolve_layer_indices(checkpoint_name),
        no_transform=no_transform,
        tissue_config=tissue_config,
        rf_overlap_override=rf_overlap,
        inf_neighborhood=inf_neighborhood,
        seed=_get_seed(checkpoint_name),
    )
    model.name = checkpoint_name or checkpoint_path.stem
    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = model.load_state_dict(checkpoint["transformed_model_state_dict"], strict=False)
    epoch = checkpoint.get("epoch")
    print(f"Loaded TopoTransformedCLIP model from {checkpoint_path} (epoch {epoch}).")
    print(msg)
    return model, epoch


def _load_videomae_model(checkpoint_name, device):
    checkpoint_path, no_transform = _resolve_vjepa_checkpoint(checkpoint_name)
    tissue_config, rf_overlap, inf_neighborhood = _resolve_tissue_config(checkpoint_name)
    model = TopoTransformedVideoMAE(
        layer_indices=_resolve_layer_indices(checkpoint_name),
        no_transform=no_transform,
        tissue_config=tissue_config,
        rf_overlap_override=rf_overlap,
        inf_neighborhood=inf_neighborhood,
        seed=_get_seed(checkpoint_name),
    )
    model.name = checkpoint_name or checkpoint_path.stem
    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = model.load_state_dict(checkpoint["transformed_model_state_dict"], strict=False)
    epoch = checkpoint.get("epoch")
    print(f"Loaded TopoTransformedVideoMAE model from {checkpoint_path} (epoch {epoch}).")
    print(msg)
    return model, epoch


def _load_llcnn_model(checkpoint_name, device):
    from models.llcnn import LLCNN_DEFAULT_CKPT

    checkpoint_path = LLCNN_DEFAULT_CKPT
    layer_name = "layer4.1"
    if checkpoint_name.startswith("llcnn.") and ":" in checkpoint_name:
        _, spec = checkpoint_name.split(":", 1)
        parts = [part for part in spec.split(",") if part]
        for part in parts:
            key, value = part.split("=", 1)
            if key == "path":
                checkpoint_path = value
            elif key == "layer":
                layer_name = value
    model = TopoTransformedLLCNN(checkpoint_path=checkpoint_path, layer_name=layer_name)
    model.name = checkpoint_name
    epoch = getattr(model.model, "epoch", None)
    print(f"Loaded TopoTransformedLLCNN model from {checkpoint_path} (epoch {epoch}).")
    return model, epoch


def _load_toponets_model(checkpoint_name, device):
    from models.toponets import TOPONETS_DEFAULT_CKPT

    checkpoint_path = TOPONETS_DEFAULT_CKPT
    layer_name = "layer4.1.conv2"
    tau = 10.0
    if checkpoint_name.startswith("toponets.") and ":" in checkpoint_name:
        _, spec = checkpoint_name.split(":", 1)
        parts = [part for part in spec.split(",") if part]
        for part in parts:
            key, value = part.split("=", 1)
            if key == "path":
                checkpoint_path = value
            elif key == "layer":
                layer_name = value
            elif key == "tau":
                tau = float(value)
    model = TopoTransformedTopoNets(checkpoint_path=checkpoint_path, layer_name=layer_name, tau=tau)
    model.name = checkpoint_name
    print(f"Loaded TopoTransformedTopoNets model tau={tau:g}, layer={layer_name}, checkpoint={checkpoint_path}.")
    return model, None


def _load_vjepa_model(checkpoint_name, device):
    checkpoint_path, no_transform = _resolve_vjepa_checkpoint(checkpoint_name)
    tissue_config, rf_overlap, inf_neighborhood = _resolve_tissue_config(checkpoint_name)
    model = TopoTransformedVJEPA(
        layer_indices=_resolve_layer_indices(checkpoint_name),
        no_transform=no_transform,
        tissue_config=tissue_config,
        rf_overlap_override=rf_overlap,
        inf_neighborhood=inf_neighborhood,
        seed=_get_seed(checkpoint_name),
    )
    model.name = checkpoint_name or checkpoint_path.stem
    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = model.load_state_dict(checkpoint["transformed_model_state_dict"], strict=False)
    epoch = checkpoint.get("epoch")
    print(f"Loaded TopoTransformedVJEPA model from {checkpoint_path} (epoch {epoch}).")
    print(msg)
    return model, epoch


def load_transformed_model(checkpoint_name, device="cuda"):
    """Load a trained transformed model while preserving existing checkpoint naming."""

    if _is_tdann_checkpoint(checkpoint_name):
        model, epoch = _load_tdann_model(checkpoint_name)
    elif _is_swapopt_checkpoint(checkpoint_name):
        model, epoch = _load_swapopt_model(checkpoint_name)
    elif _is_som_checkpoint(checkpoint_name):
        model, epoch = _load_som_model(checkpoint_name, device)
    elif _is_clip_checkpoint(checkpoint_name):
        model, epoch = _load_clip_model(checkpoint_name, device)
    elif _is_videomae_checkpoint(checkpoint_name):
        model, epoch = _load_videomae_model(checkpoint_name, device)
    elif _is_llcnn_checkpoint(checkpoint_name):
        model, epoch = _load_llcnn_model(checkpoint_name, device)
    elif _is_toponets_checkpoint(checkpoint_name):
        model, epoch = _load_toponets_model(checkpoint_name, device)
    else:
        model, epoch = _load_vjepa_model(checkpoint_name, device)

    model.to(device)
    return model, epoch


__all__ = [
    "load_transformed_model",
    "validate_autocorr",
    "validate_floc",
    "validate_floc_human",
    "validate_invertibility",
]


if __name__ == "__main__":
    model, epoch = load_transformed_model("tdann/model_final_checkpoint_phase199_seed1.torch", device='cpu')
