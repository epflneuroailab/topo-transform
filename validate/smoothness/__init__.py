import libpysal as lp
import numpy as np
from tqdm import tqdm
from esda.moran import Moran

from .utils import compute_morans_I_neural
from .utils import compute_nsd_high_adjacency_list
from validate.floc import validate_floc
from validate.floc import validate_floc_human
from validate.rois.nsd import get_region_voxels

NSD_HIGH = get_region_voxels(
    ["high-ventral", "high-lateral", "high-dorsal"]
)
human_lh_adj_list, human_rh_adj_list = compute_nsd_high_adjacency_list()


def _split_high_region_mask(mask):
    midpoint = len(mask) // 2
    return mask[:midpoint], mask[midpoint:]


def _model_localizer_values(model, transform, dataset_name, batch_size, device, frames_per_video, video_fps, fwhm_mm, resolution_mm):
    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        model_t_val_dict = validate_floc(
            model,
            transform,
            dataset_names=[dataset_name],
            batch_size=batch_size,
            device=device,
            frames_per_video=frames_per_video,
            video_fps=video_fps,
        )[0]
        model_sheet_dims = model.smoothed_layer_positions[0].dims
        model_w = lp.weights.lat2W(*model_sheet_dims[-2:], rook=False)
        model_t_val_dict = {key: values[0] for key, values in model_t_val_dict.items()}
    return model_t_val_dict, model_w


def _average_hemisphere_smoothness(lh_values, rh_values):
    lh_smoothness = compute_morans_I_neural(lh_values, human_lh_adj_list)
    rh_smoothness = compute_morans_I_neural(rh_values, human_rh_adj_list)
    return (lh_smoothness + rh_smoothness) / 2.0

def validate_smoothness(
        model,
        transform,
        dataset_name,
        batch_size=32,
        device='cuda',
        frames_per_video=24,
        video_fps=12,
        fwhm_mm=2.0,
        resolution_mm=1.0,
    ):

    print("Assuming single sheet and smoothing...")
    model_t_val_dict, model_w = _model_localizer_values(
        model,
        transform,
        dataset_name,
        batch_size,
        device,
        frames_per_video,
        video_fps,
        fwhm_mm,
        resolution_mm,
    )
    model_cates = list(model_t_val_dict.keys())

    human_t_val_dict = validate_floc_human(dataset_names=[dataset_name])[0]
    human_lh_t_val_dict, human_rh_t_val_dict = _filter_high_regions(human_t_val_dict)
    human_cates = list(human_lh_t_val_dict.keys())

    common_cates = list(set(model_cates) & set(human_cates))
    print(f"Common categories: {common_cates}")
    smoothness_results = {}
    for cate in common_cates:
        model_t_vals = model_t_val_dict[cate]
        human_lh_t_vals = human_lh_t_val_dict[cate]
        human_rh_t_vals = human_rh_t_val_dict[cate]

        model_smoothness = _moran_ignore_nans(model_t_vals, model_w)
        human_smoothness = _average_hemisphere_smoothness(human_lh_t_vals, human_rh_t_vals)

        print(f"Cate: {cate}, Model smoothness: {model_smoothness:.4f}, Human smoothness: {human_smoothness:.4f}")

        smoothness_results[cate] = {
            'model_smoothness': model_smoothness,
            'human_smoothness': human_smoothness,
        }

    return smoothness_results

def _moran_ignore_nans(values, w):
    mask = np.isfinite(values)
    if mask.all():
        return Moran(values, w).I
    kept = np.flatnonzero(mask)
    if kept.size < 2:
        return np.nan
    old_to_new = {int(old): new for new, old in enumerate(kept)}
    neighbors = {}
    weights = {}
    for old_idx in kept:
        old_idx = int(old_idx)
        nbrs = []
        wts = []
        for nbr, wt in zip(w.neighbors.get(old_idx, []), w.weights.get(old_idx, [])):
            if nbr in old_to_new:
                nbrs.append(old_to_new[nbr])
                wts.append(wt)
        neighbors[old_to_new[old_idx]] = nbrs
        weights[old_to_new[old_idx]] = wts
    w_sub = lp.weights.W(neighbors, weights, silence_warnings=True)
    return Moran(values[mask], w_sub).I

def _filter_high_regions(t_val_dict):
    sel = NSD_HIGH
    sel_lh, sel_rh = _split_high_region_mask(sel)
    lh_t_val_dict = {}
    rh_t_val_dict = {}
    for cat_name in t_val_dict:
        t_vals = t_val_dict[cat_name]
        lh_t_vals = t_vals[:len(t_vals)//2][sel_lh]
        rh_t_vals = t_vals[len(t_vals)//2:][sel_rh]
        lh_t_val_dict[cat_name] = lh_t_vals
        rh_t_val_dict[cat_name] = rh_t_vals
    return lh_t_val_dict, rh_t_val_dict

def compute_activity_smoothness_neural(activity):
    sel = NSD_HIGH
    sel_lh, sel_rh = _split_high_region_mask(sel)
    lh_activity = activity[:, :len(sel)//2][:, sel_lh]
    rh_activity = activity[:, len(sel)//2:][:, sel_rh]
    B = lh_activity.shape[0]
    smoothness = []
    for b in range(B):
        smoothness.append(_average_hemisphere_smoothness(lh_activity[b], rh_activity[b]))
    return np.array(smoothness)

def compute_activity_smoothness_model(activity, model):
    import torch
    from libpysal import weights
    model_sheet_dims = model.smoothed_layer_positions[0].dims

    if isinstance(activity, torch.Tensor):
        activity = activity.cpu().numpy()

    B = activity.shape[0]
    model_w = weights.lat2W(*model_sheet_dims[-2:])
    smoothness = []
    for b in tqdm(range(B), desc="Computing model smoothness"):
        s = _moran_ignore_nans(activity[b], model_w)
        smoothness.append(s)
    return np.array(smoothness)


__all__ = [
    "NSD_HIGH",
    "compute_activity_smoothness_model",
    "compute_activity_smoothness_neural",
    "validate_smoothness",
]
