from collections import OrderedDict

import config

from .common import (
    CLIP_CKPTS,
    VIDEOMAE_CKPTS,
    MODEL_CKPTS,
    SOM_CKPTS,
    TDANN_CKPTS,
    UNOPTIMIZED_CKPTS,
    SWAPOPT_CKPTS,
    ONELAYER_CKPTS,
    MODEL_C,
    DEFAULT_C,
)
from .get_localizers import localizers


def _existing_checkpoints(checkpoint_names):
    checkpoint_dir = config.CACHE_DIR / "checkpoints"
    return [name for name in checkpoint_names if (checkpoint_dir / name).exists()]


CKPT_GROUPS = OrderedDict(
    [
        ("TopoTransform", MODEL_CKPTS),
        ("TDANN", TDANN_CKPTS),
        ("UNOPTIMIZED", UNOPTIMIZED_CKPTS),
        ("SWAPOPT", SWAPOPT_CKPTS),
        ("ONELAYER", ONELAYER_CKPTS),
    ]
)

SOM_AVAILABLE_CKPTS = _existing_checkpoints(SOM_CKPTS)
if SOM_AVAILABLE_CKPTS:
    CKPT_GROUPS["SOM"] = SOM_AVAILABLE_CKPTS
CLIP_AVAILABLE_CKPTS = _existing_checkpoints(CLIP_CKPTS)
if CLIP_AVAILABLE_CKPTS:
    CKPT_GROUPS["CLIP"] = CLIP_AVAILABLE_CKPTS
VIDEOMAE_AVAILABLE_CKPTS = _existing_checkpoints(VIDEOMAE_CKPTS)
if VIDEOMAE_AVAILABLE_CKPTS:
    CKPT_GROUPS["VideoMAE"] = VIDEOMAE_AVAILABLE_CKPTS

DEFAULT_METHOD_ORDER = ("TopoTransform", "CLIP", "VideoMAE", "SOM", "TDANN", "SWAPOPT", "UNOPTIMIZED")
FULL_METHOD_ORDER = ("TopoTransform", "CLIP", "VideoMAE", "SOM", "TDANN", "SWAPOPT", "UNOPTIMIZED", "ONELAYER")

METHOD_LABELS = {
    "TopoTransform": "TopoTransform",
    "CLIP": "CLIP",
    "VideoMAE": "VideoMAE",
    "TDANN": "TDANN",
    "UNOPTIMIZED": "VJEPA",
    "SWAPOPT": "SwapOpt",
    "ONELAYER": "OneLayer",
    "SOM": "SOM",
}

METHOD_COLORS = {
    "TopoTransform": MODEL_C,
    "CLIP": "#1B9E77",
    "VideoMAE": "#7570B3",
    "SOM": "#D95F02",
    "TDANN": DEFAULT_C,
    "UNOPTIMIZED": DEFAULT_C,
    "SWAPOPT": DEFAULT_C,
    "ONELAYER": DEFAULT_C,
}


def resolve_group_names(group_names=None, default=DEFAULT_METHOD_ORDER):
    """Return the subset of requested checkpoint groups that exists."""
    if group_names is None:
        group_names = default
    if isinstance(group_names, str):
        group_names = (group_names,)
    return [name for name in group_names if name in CKPT_GROUPS]


def get_ckpt_groups(group_names=None, default=DEFAULT_METHOD_ORDER):
    return {name: CKPT_GROUPS[name] for name in resolve_group_names(group_names, default=default)}


def collect_by_ckpt(ckpt_names, fn, *args, verbose=False, prefix="Processing checkpoint: ", **kwargs):
    results = []
    for ckpt_name in ckpt_names:
        if verbose:
            print(f"{prefix}{ckpt_name}")
        results.append(fn(ckpt_name, *args, **kwargs))
    return results


def collect_group_results(group_names, fn, first_kwargs=None, rest_kwargs=None):
    results = {}
    for i, group_name in enumerate(group_names):
        kwargs = first_kwargs if i == 0 else rest_kwargs
        results[group_name] = fn(CKPT_GROUPS[group_name], **(kwargs or {}))
    return results


def collect_localizer_tvals(
    ckpt_names,
    dataset="robert",
    ret_merged=True,
    verbose=True,
    on_result=None,
    **localizer_kwargs,
):
    def _load(ckpt_name):
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(
            ckpt_name,
            ret_merged=ret_merged,
            **localizer_kwargs,
        )
        if on_result is not None:
            on_result(ckpt_name, t_vals_dicts, p_vals_dicts, layer_positions)
        return t_vals_dicts[dataset]

    return collect_by_ckpt(ckpt_names, _load, verbose=verbose)
