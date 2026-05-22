from .utils import visualize_all_rois_v2
from .utils import visualize_patches
from .utils import visualize_tvals

from .categories import localize_categories
from .temporal import localize_temporal
from .motion import localize_motion
from .v6 import localize_v6
from .psts import localize_psts
from .pitcher import localize_pitcher, localize_pitcher_human
from .robert import localize_robert, load_robert_tvals, localize_robert_human
from .afraz import localize_afraz
from .konkle import localize_konkle, localize_konkle_animacy, localize_konkle_big_small
from .registry import FLOC_DATASETS
from .registry import get_human_localizer_runner
from .registry import get_model_localizer_runner


def _get_layer_positions(model):
    if model.smoothing:
        return [lp.coordinates.cpu() for lp in model.smoothed_layer_positions]
    return [lp.coordinates.cpu() for lp in model.layer_positions]


def validate_floc(
        model, 
        transform, 
        dataset_names, 
        epoch=None, 
        viz_dir=None, 
        viz_patches=False, 
        viz_params=None,
        batch_size=32, 
        device='cuda',
        frames_per_video=24,
        video_fps=12,
        plot_individual=False,
        plot_aggregate=False,
    ):

    if viz_params is None:
        viz_params = {}

    if viz_dir is not None and epoch is None:
        viz_dir = viz_dir / "eval"
        viz_dir.mkdir(parents=True, exist_ok=True)

    layer_positions = _get_layer_positions(model)

    t_vals_dicts = []
    for dataset_name in dataset_names:
        runner = get_model_localizer_runner(dataset_name)
        t_vals_dict = runner(
            model,
            transform,
            batch_size=batch_size,
            device=device,
            frames_per_video=frames_per_video,
            video_fps=video_fps,
        )

        t_vals_dicts.append(t_vals_dict)

        if viz_dir is not None and plot_individual:
            suffix = f"_{epoch + 1}" if epoch is not None else ""
            visualize_tvals(t_vals_dict, layer_positions, viz_dir, prefix=f"{dataset_name}_", suffix=suffix, **viz_params)
            if viz_patches:
                visualize_patches(t_vals_dict, layer_positions, viz_dir, prefix=f"{dataset_name}_", suffix=f"_patches{suffix}")

    if viz_dir is not None and plot_aggregate:
        suffix = f"_{epoch + 1}" if epoch is not None else ""
        visualize_all_rois_v2(t_vals_dicts, layer_positions, viz_dir, prefix="rois_", suffix=suffix)

    return t_vals_dicts


def validate_floc_human(
    dataset_names,
):
    t_vals_dicts = []
    for dataset_name in dataset_names:
        t_vals_dict = get_human_localizer_runner(dataset_name)()
        t_vals_dicts.append(t_vals_dict)
    return t_vals_dicts


__all__ = [
    "FLOC_DATASETS",
    "localize_afraz",
    "localize_categories",
    "localize_konkle",
    "localize_konkle_animacy",
    "localize_konkle_big_small",
    "localize_motion",
    "localize_pitcher",
    "localize_pitcher_human",
    "localize_psts",
    "localize_robert",
    "localize_robert_human",
    "localize_temporal",
    "localize_v6",
    "get_human_localizer_runner",
    "get_model_localizer_runner",
    "validate_floc",
    "validate_floc_human",
]
