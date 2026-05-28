from pathlib import Path
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from data import Kinetics400
from topo import TopoTransformedVJEPA, TopoTransformedTDANN

from utils import cached
from validate import load_transformed_model
from models import vit_transform

def _get_seed(ckpt_name: str) -> int:
    match = re.search(r"sd(\d+)", ckpt_name) or re.search(r"seed(\d+)", ckpt_name)
    if not match:
        raise ValueError(f"Could not extract seed from {ckpt_name}")
    return int(match.group(1))


def _resolve_layer_indices(ckpt_name: str) -> list[int]:
    match = re.search(r"vjepa_([0-9_]+)_single", ckpt_name) or re.search(r"vjepa_([0-9_]+)_", ckpt_name)
    if match:
        return [int(index) for index in match.group(1).split("_")]
    return [14, 18, 22] if "14_18_22" in ckpt_name else [18]


def _resolve_tissue_config(ckpt_name: str):
    tissue_config = "small" if "_small" in ckpt_name else "vtc"
    match = re.search(r"_rf([0-9p]+)", ckpt_name)
    rf_overlap = float(match.group(1).replace("p", ".")) if match else None
    inf_neighborhood = "_neighbInf" in ckpt_name and tissue_config != "small"
    return tissue_config, rf_overlap, inf_neighborhood


def _load_vjepa_with_single_sheet_override(ckpt_name: str, device: str, single_sheet: bool):
    checkpoint_path = config.CACHE_DIR / "checkpoints" / ckpt_name.replace("unoptimized.", "")
    tissue_config, rf_overlap, inf_neighborhood = _resolve_tissue_config(ckpt_name)
    model = TopoTransformedVJEPA(
        layer_indices=_resolve_layer_indices(ckpt_name),
        no_transform=ckpt_name.startswith("unoptimized."),
        single_sheet=single_sheet,
        tissue_config=tissue_config,
        rf_overlap_override=rf_overlap,
        inf_neighborhood=inf_neighborhood,
        seed=_get_seed(ckpt_name),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["transformed_model_state_dict"]
    state = {
        key: value
        for key, value in state.items()
        if "coordinates" not in key and "neighborhood" not in key
    }
    msg = model.load_state_dict(state, strict=False)
    print(f"Loaded TopoTransformedVJEPA model from {checkpoint_path} with single_sheet={single_sheet}.")
    print(msg)
    return model


def _validate_features(ckpt_name, single_sheet=None):

    is_swapopt = ckpt_name == "swapopt"

    # Load data
    data = Kinetics400(train_transforms=vit_transform, test_transforms=vit_transform, fps=12)
    val_loader = DataLoader(data.valset, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if single_sheet is None:
        model = load_transformed_model(ckpt_name, device=device)[0]
    else:
        model = _load_vjepa_with_single_sheet_override(ckpt_name, device, single_sheet=single_sheet)
        model.to(device)
    model.eval()

    # Extract features
    all_features = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Extracting features'):
            videos = batch[0].to(device)
            features, positions = model(videos)
            all_features.append([f.cpu() for f in features])
    all_features = [torch.cat([batch[l] for batch in all_features], dim=0) for l in range(len(all_features[0]))]
    return all_features, positions

def _cache_key(ckpt_name: str, single_sheet=None) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(ckpt_name).stem)
    if single_sheet is None:
        return f"validate_features_{safe_name}"
    sheet_name = "single" if single_sheet else "multi"
    return f"validate_features_{safe_name}_{sheet_name}"


def validate_features(ckpt_name: str, single_sheet=None):
    return cached(_cache_key(ckpt_name, single_sheet=single_sheet), rerun=False)(_validate_features)(ckpt_name, single_sheet=single_sheet)
