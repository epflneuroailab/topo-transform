from utils import cached
from models import clip_transform
from models import llcnn_transform
from models import vit_transform
from validate.floc.registry import LOCALIZER_DATASETS
from validate.floc.registry import get_model_localizer_runner
from validate import load_transformed_model
from validate.correction import fwe

from .common import LOCALIZER_P_THRESHOLD
from .common import LOCALIZER_T_THRESHOLD
from .localizer_registry import get_human_localizer_mask
from .localizer_registry import get_localizer_result_key
from .localizer_registry import get_roi_t_threshold

FLOC_DATASETS = LOCALIZER_DATASETS
LOCALIZER_RERUN = False

def _get_layer_positions(model):
    if model.smoothing:
        return [lp.coordinates.cpu() for lp in model.smoothed_layer_positions]
    return [lp.coordinates.cpu() for lp in model.layer_positions]


def _get_input_transform(model):
    if model.__class__.__name__ == "TopoTransformedLLCNN":
        return llcnn_transform
    return clip_transform if model.__class__.__name__ == "TopoTransformedCLIP" else vit_transform


def _merge_dict_list(dicts):
    merged = {}
    for values in dicts:
        merged.update(values)
    return merged


def _merged_localizers(*args, **kwargs):
    t_vals_dicts, p_vals_dicts, layer_positions = localizers(*args, ret_merged=False, **kwargs)
    return _merge_dict_list(t_vals_dicts), _merge_dict_list(p_vals_dicts), layer_positions


def _resolve_t_and_p_vals(roi, t_val_dict, p_val_dict):
    key = get_localizer_result_key(roi)
    return t_val_dict[key], p_val_dict[key]


def _correct_p_values(p_vals_dicts):
    corrected = []
    for p_vals_dict in p_vals_dicts:
        corrected_dict = {}
        for roi_name, p_vals in p_vals_dict.items():
            shapes = [p_val.shape for p_val in p_vals]
            corrected_vals = [fwe(p_val) for p_val in p_vals]
            corrected_dict[roi_name] = [
                corrected_val.reshape(shape)
                for corrected_val, shape in zip(corrected_vals, shapes)
            ]
        corrected.append(corrected_dict)
    return corrected

def _localizers(
        checkpoint_name, 
        dataset_names, 
        device='cuda',
        batch_size=16,
        video_fps=12,
        frames_per_video=24,
        fwhm_mm=2.0,
        resolution_mm=1.0,
    ):

    model, epoch = load_transformed_model(checkpoint_name=checkpoint_name, device=device)
    model.eval()
    transform = _get_input_transform(model)

    is_swapopt = "swapopt" in checkpoint_name
    frames_per_video = 24 if not is_swapopt else 16

    with model.smoothing_enabled(
            fwhm_mm=fwhm_mm, 
            resolution_mm=resolution_mm, 
        ):

        layer_positions = _get_layer_positions(model)

        t_vals_dicts = []
        p_vals_dicts = []
        for dataset_name in dataset_names:
            runner = get_model_localizer_runner(dataset_name)
            t_vals_dict, p_vals_dict = runner(
                model,
                transform,
                ret_pvals=True,
                batch_size=batch_size,
                device=device,
                frames_per_video=frames_per_video,
                video_fps=video_fps,
            )

            t_vals_dicts.append(t_vals_dict)
            p_vals_dicts.append(p_vals_dict)

        return t_vals_dicts, p_vals_dicts, layer_positions


def localizers(
        checkpoint_name, 
        dataset_names=FLOC_DATASETS, 
        device='cuda', 
        frames_per_video=24, 
        video_fps=12,
        fwhm_mm=2.0,
        resolution_mm=1.0,
        ret_merged=False,
    ):
    import hashlib
    dataset_str = '_'.join(sorted(dataset_names))
    hash_suffix = hashlib.md5(dataset_str.encode()).hexdigest()[:8]
    t_vals_dicts, p_vals_dicts, layer_positions = cached(
        f"localizers_{checkpoint_name}_{hash_suffix}_{fwhm_mm}_{resolution_mm}",
        rerun=LOCALIZER_RERUN
    )(_localizers)(checkpoint_name, dataset_names, device=device, frames_per_video=frames_per_video, video_fps=video_fps, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    # p val correction
    p_vals_dicts = _correct_p_values(p_vals_dicts)

    if ret_merged:
        t_vals_dicts = _merge_dict_list(t_vals_dicts)
        p_vals_dicts = _merge_dict_list(p_vals_dicts)

    return t_vals_dicts, p_vals_dicts, layer_positions


def get_localizer_model(rois, ckpt_name, p_thres=LOCALIZER_P_THRESHOLD, t_thres=LOCALIZER_T_THRESHOLD, fwhm_mm=2.0, resolution_mm=1.0):
    t_val_dict, p_val_dict, layer_positions = _merged_localizers(
        ckpt_name,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
    )

    def _filter(roi, p_val, t_val):
        t_thres_used = get_roi_t_threshold(roi, t_thres)
        mask = (p_val < p_thres) & (t_val > t_thres_used)
        return mask

    ret = []
    for roi in rois:
        t_vals, p_vals = _resolve_t_and_p_vals(roi, t_val_dict, p_val_dict)
        masks = [_filter(roi, p_val, t_val) for p_val, t_val in zip(p_vals, t_vals)]  # layers
        ret.append(masks)
    return ret


def get_localizer_human(rois):
    return [get_human_localizer_mask(roi) for roi in rois]


__all__ = [
    "FLOC_DATASETS",
    "LOCALIZER_RERUN",
    "get_localizer_human",
    "get_localizer_model",
    "localizers",
]
