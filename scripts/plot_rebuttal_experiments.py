import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import CACHE_DIR
from config import PLOTS_DIR
from .common import HUMAN_C
from .common import LOCALIZER_P_THRESHOLD
from .common import LOCALIZER_T_THRESHOLD
from .common import all_roi_colors
from .get_localizers import localizers
from .localizer_registry import get_localizer_result_key
from .localizer_registry import get_roi_t_threshold
from .plot_localizers import plot_all_rois
from .plot_utils import ensure_dir
from .plot_utils import to_numpy
from .analysis_utils import CKPT_GROUPS
from .analysis_utils import METHOD_LABELS
from .get_smoothness import smoothness
from .plot_localizer_decode import localizer_decode_clustered
from .get_localizer_decode_ceiling import localizer_decode_ceiling
from .plot_localizer_motion import plot_all_rois as plot_motion_rois
from .plot_localizer_motion import collect_localizer_tvals
from validate.floc.utils.cluster import find_patches


# Keep the rebuttal plots focused and fast: this tests ventral fLoc-style
# organization, which is the direct target of the CLIP/static-image question.
LOCALIZER_DATASETS = ["vpnl"]
HALF_RELIABILITY_DATASETS = ["vpnl", "pitzalis", "biomotion"]
ROI_ORDER = ["face", "body", "place", "object"]
HALF_RELIABILITY_ROIS = ["face", "body", "place", "mt", "v6", "psts"]
ROI_LABELS = {
    "face": "Face",
    "body": "Body",
    "place": "Place",
    "object": "Object",
    "v6": "V6",
    "psts": "pSTS",
    "mt": "MT",
}
SOM_ROI_ORDER = ["face", "body", "place"]
SOM_TOP25_T_THRESHOLDS = {
    "face": 7.340281963348389,
    "body": -8.004393577575684,
    "place": 17.77458953857422,
}
SOM_TOP25_P_THRESHOLD = 1.0
REBUTTAL_DECODE_ROIS = ["face", "place", "body", "v6", "psts"]
REBUTTAL_CLIP_DECODE_ROIS = ["face", "place", "body"]
REBUTTAL_MOTION_ROIS = ["face", "body", "place", "mt", "v6", "psts"]
REBUTTAL_METHOD_ORDER = [
    "UNOPTIMIZED",
    "SOM",
    "TopoTransform",
    "CLIP_RAW",
    "CLIP",
    "LLCNN",
    "TOPONETS",
    "TDANN",
    "SWAPOPT",
]
REBUTTAL_METHOD_LABELS = {
    "UNOPTIMIZED": "VJEPA",
    "SOM": "VJEPA-SOM",
    "TopoTransform": "VJEPA-TopoTransform",
    "CLIP_RAW": "CLIP",
    "CLIP": "CLIP-TopoTransform",
    "LLCNN": "LLCNN",
    "TOPONETS": "TopoNets",
    "TDANN": "TDANN",
    "SWAPOPT": "VJEPA-SwapOpt",
}
REBUTTAL_METHOD_COLORS = {
    "UNOPTIMIZED": "#9A9A9A",
    "CLIP_RAW": "#9A9A9A",
    "LLCNN": "#9A9A9A",
    "TOPONETS": "#9A9A9A",
    "TopoTransform": "#1B9E77",
    "CLIP": "#1B9E77",
    "SOM": "#9A9A9A",
    "TDANN": "#9A9A9A",
    "SWAPOPT": "#9A9A9A",
}
REBUTTAL_ADD_MODEL_DROP_GROUPS = {"UNOPTIMIZED", "CLIP_RAW", "SOM"}
REBUTTAL_VJEPA_ONLY_GROUPS = ["UNOPTIMIZED", "SOM", "TopoTransform"]
REBUTTAL_BASE_TOPO_GROUPS = [
    "UNOPTIMIZED",
    "TopoTransform",
    "CLIP_RAW",
    "CLIP",
    "TDANN",
    "LLCNN",
    "TOPONETS",
    "SWAPOPT",
]

CLIP_CKPTS = [
    "best_transformed_model_global_clip_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd41.pt",
    "best_transformed_model_global_clip_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt",
]
VJEPA_CKPT = "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt"
VIDEOMAE_CKPTS = [
    "best_transformed_model_global_videomae_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd41.pt",
    "best_transformed_model_global_videomae_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt",
]
HALF_CKPTS = {
    "half A": "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42_half_a.pt",
    "half B": "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42_half_b.pt",
}
SMALL_CKPT = "best_transformed_model_global_vjepa_4_single_small_rf0p1_kinetics400_lr1e-4_bs32_sd42.pt"
SOM_CKPT = "som_vjepa_22_single_1mm_kinetics400_bs32_sd42.pt"
LLCNN_CKPT = "llcnn.resnet18contopo_gaussian0p23_epoch100_layer4p1"
TOPONETS_CKPT = "toponets.resnet18_tau10_layer4p1conv2"
ADD_MODEL_KEYS = {"tdann", "llcnn", "toponets", "swapopt"}


def _exists(ckpt_name):
    return (CACHE_DIR / "checkpoints" / ckpt_name).exists()


def _unoptimized(ckpt_name):
    return f"unoptimized.{ckpt_name}"


def _ckpt_available(ckpt_name):
    if ckpt_name.startswith("swapopt_single_sheet_seed"):
        seed = ckpt_name.split("swapopt_single_sheet_seed", 1)[1].split(".")[0]
        return (CACHE_DIR / "positions" / "swapopt_single_sheet" / f"seed{seed}" / "single_sheet.npz").exists()
    if ckpt_name.startswith("tdann."):
        return (CACHE_DIR / "checkpoints" / ckpt_name.replace("tdann.", "tdann/")).exists()
    if ckpt_name.startswith("llcnn."):
        from models.llcnn import LLCNN_DEFAULT_CKPT

        return LLCNN_DEFAULT_CKPT.exists()
    if ckpt_name.startswith("toponets."):
        from models.toponets import TOPONETS_DEFAULT_CKPT, TOPONETS_ROOT

        return TOPONETS_DEFAULT_CKPT.exists() or TOPONETS_ROOT.exists()
    if ckpt_name.startswith("unoptimized."):
        ckpt_name = ckpt_name.replace("unoptimized.", "")
    return _exists(ckpt_name)


def _normalize_add_models(add_models=None):
    if add_models is None:
        return set()
    if isinstance(add_models, str):
        add_models = [add_models]
    normalized = {model.lower() for model in add_models}
    if not normalized or "all" in normalized:
        return set(ADD_MODEL_KEYS)
    return normalized


def _existing_rebuttal_groups(max_clip_seeds=None, add_models=None):
    add_models = _normalize_add_models(add_models)
    clip_ckpts = CLIP_CKPTS[:max_clip_seeds] if max_clip_seeds is not None else CLIP_CKPTS
    groups = {
        "UNOPTIMIZED": [ckpt for ckpt in CKPT_GROUPS.get("UNOPTIMIZED", []) if _ckpt_available(ckpt)],
        "SOM": [SOM_CKPT] if _ckpt_available(SOM_CKPT) else [],
        "TopoTransform": [ckpt for ckpt in CKPT_GROUPS.get("TopoTransform", []) if _ckpt_available(ckpt)],
        "CLIP_RAW": [_unoptimized(ckpt) for ckpt in clip_ckpts if _ckpt_available(ckpt)],
        "CLIP": [ckpt for ckpt in clip_ckpts if _ckpt_available(ckpt)],
        "LLCNN": [LLCNN_CKPT] if "llcnn" in add_models and _ckpt_available(LLCNN_CKPT) else [],
        "TOPONETS": [TOPONETS_CKPT] if "toponets" in add_models and _ckpt_available(TOPONETS_CKPT) else [],
        "TDANN": [ckpt for ckpt in CKPT_GROUPS.get("TDANN", []) if _ckpt_available(ckpt)],
        "SWAPOPT": [ckpt for ckpt in CKPT_GROUPS.get("SWAPOPT", []) if _ckpt_available(ckpt)],
    }
    return {group: groups[group] for group in REBUTTAL_METHOD_ORDER if groups.get(group)}


def _load_localizers(ckpt_name, device, fwhm_mm, resolution_mm, dataset_names=None):
    if dataset_names is None:
        dataset_names = LOCALIZER_DATASETS
    return localizers(
        ckpt_name,
        dataset_names=dataset_names,
        device=device,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
        ret_merged=True,
    )


def _roi_values(roi, t_vals_dict, p_vals_dict):
    key = get_localizer_result_key(roi)
    return t_vals_dict[key], p_vals_dict[key]


def _roi_mask(roi, t_vals, p_vals, t_thresholds=None, p_threshold=LOCALIZER_P_THRESHOLD):
    t_threshold = (
        t_thresholds[roi]
        if t_thresholds is not None and roi in t_thresholds
        else get_roi_t_threshold(roi, LOCALIZER_T_THRESHOLD)
    )
    masks = []
    for t_val, p_val in zip(t_vals, p_vals):
        masks.append(
            (np.asarray(p_val).reshape(-1) < p_threshold)
            & (np.asarray(t_val).reshape(-1) > t_threshold)
        )
    return np.concatenate(masks, axis=0)


def _roi_positions(layer_positions):
    return np.concatenate([to_numpy(pos).reshape(-1, 2) for pos in layer_positions], axis=0)


def _largest_cluster_fraction(
    roi,
    t_vals,
    p_vals,
    layer_positions,
    t_thresholds=None,
    p_threshold=LOCALIZER_P_THRESHOLD,
):
    t_threshold = (
        t_thresholds[roi]
        if t_thresholds is not None and roi in t_thresholds
        else get_roi_t_threshold(roi, LOCALIZER_T_THRESHOLD)
    )
    total_area = 0
    largest_area = 0
    n_significant = 0
    for t_val, p_val, pos in zip(t_vals, p_vals, layer_positions):
        n_significant += int(
            np.sum(
                (np.asarray(p_val).reshape(-1) < p_threshold)
                & (np.asarray(t_val).reshape(-1) > t_threshold)
            )
        )
        patches = find_patches(
            positions=pos,
            selectivities=np.asarray(t_val),
            p_values=np.asarray(p_val),
            t_threshold=t_threshold,
            p_threshold=p_threshold,
        )
        areas = [patch.area for patch in patches]
        total_area += sum(areas)
        largest_area = max([largest_area] + areas)
    if total_area == 0:
        return 0.0 if n_significant > 0 else np.nan
    return largest_area / total_area


def _mask_centroid(mask, positions):
    if not np.any(mask):
        return np.array([np.nan, np.nan])
    return positions[mask].mean(axis=0)


def _dice(a, b):
    denom = int(np.sum(a)) + int(np.sum(b))
    if denom == 0:
        return np.nan
    return 2.0 * int(np.sum(a & b)) / denom


def collect_model_clustering(device, fwhm_mm, resolution_mm, include_videomae=False, max_clip_seeds=None):
    rows = []
    clip_ckpts = CLIP_CKPTS[:max_clip_seeds] if max_clip_seeds is not None else CLIP_CKPTS
    model_groups = {
        "VJEPA raw": [_unoptimized(VJEPA_CKPT)] if _exists(VJEPA_CKPT) else [],
        "VJEPA + TopoTransform": [VJEPA_CKPT] if _exists(VJEPA_CKPT) else [],
        "CLIP raw": [_unoptimized(ckpt) for ckpt in clip_ckpts if _exists(ckpt)],
        "CLIP + TopoTransform": [ckpt for ckpt in clip_ckpts if _exists(ckpt)],
        "SOM": [SOM_CKPT] if _exists(SOM_CKPT) else [],
    }
    if include_videomae:
        model_groups.update(
            {
                "VideoMAE raw": [_unoptimized(ckpt) for ckpt in VIDEOMAE_CKPTS if _exists(ckpt)],
                "VideoMAE + TopoTransform": [ckpt for ckpt in VIDEOMAE_CKPTS if _exists(ckpt)],
            }
        )

    for group, ckpts in model_groups.items():
        roi_order = SOM_ROI_ORDER if group == "SOM" else ROI_ORDER
        t_thresholds = SOM_TOP25_T_THRESHOLDS if group == "SOM" else None
        p_threshold = SOM_TOP25_P_THRESHOLD if group == "SOM" else LOCALIZER_P_THRESHOLD
        for ckpt in ckpts:
            print(f"[model] {group}: {ckpt}")
            t_vals_dict, p_vals_dict, layer_positions = _load_localizers(
                ckpt, device, fwhm_mm, resolution_mm
            )
            for roi in roi_order:
                t_vals, p_vals = _roi_values(roi, t_vals_dict, p_vals_dict)
                rows.append(
                    {
                        "group": group,
                        "ckpt": ckpt,
                        "roi": roi,
                        "roi_label": ROI_LABELS[roi],
                        "largest_cluster_fraction": _largest_cluster_fraction(
                            roi,
                            t_vals,
                            p_vals,
                            layer_positions,
                            t_thresholds=t_thresholds,
                            p_threshold=p_threshold,
                        ),
                        "n_significant_units": int(
                            np.sum(
                                _roi_mask(
                                    roi,
                                    t_vals,
                                    p_vals,
                                    t_thresholds=t_thresholds,
                                    p_threshold=p_threshold,
                                )
                            )
                        ),
                        "peak_t": float(
                            np.nanmax(
                                [
                                    np.nanmax(np.asarray(t_val))
                                    for t_val in t_vals
                                ]
                            )
                        ),
                        "t_threshold": (
                            t_thresholds[roi]
                            if t_thresholds is not None and roi in t_thresholds
                            else get_roi_t_threshold(roi, LOCALIZER_T_THRESHOLD)
                        ),
                        "p_threshold": p_threshold,
                    }
                )
    return pd.DataFrame(rows)


def _moving_mean_from_smoothness(result):
    moving = [name for name in result if "moving" in name.lower()]
    if not moving:
        moving = list(result.keys())
    return float(np.mean([result[name]["model_smoothness"] for name in moving]))


def _human_moving_mean_from_smoothness(result):
    moving = [name for name in result if "moving" in name.lower()]
    if not moving:
        moving = list(result.keys())
    return float(np.mean([result[name]["human_smoothness"] for name in moving]))


def _decode_rois_for_group(group):
    if group in {"CLIP_RAW", "CLIP", "LLCNN", "TOPONETS", "TDANN"}:
        return REBUTTAL_CLIP_DECODE_ROIS
    return REBUTTAL_DECODE_ROIS


def _decode_run_rois_for_group(group):
    if group in {"LLCNN", "TOPONETS"}:
        return REBUTTAL_CLIP_DECODE_ROIS
    return REBUTTAL_DECODE_ROIS


def collect_existing_experiment_metrics(
    fwhm_mm=2.0,
    resolution_mm=1.0,
    max_clip_seeds=None,
    add_models=None,
    only_groups=None,
):
    """Summarize established analyses used elsewhere in the repo.

    This intentionally avoids the earlier ad hoc cluster-fraction metric.
    """
    rows = []
    method_groups = _existing_rebuttal_groups(max_clip_seeds=max_clip_seeds, add_models=add_models)
    if only_groups is not None:
        only_groups = set(only_groups)
        method_groups = {
            group: ckpts
            for group, ckpts in method_groups.items()
            if group in only_groups
        }

    for group, ckpts in method_groups.items():
        label = REBUTTAL_METHOD_LABELS.get(group, METHOD_LABELS.get(group, group))
        print(f"[existing] {label}: {len(ckpts)} checkpoint(s)")

        for ckpt in ckpts:
            decode_rois = _decode_rois_for_group(group)
            decode_run_rois = _decode_run_rois_for_group(group)
            decode_scores = localizer_decode_clustered(
                ckpt,
                decode_run_rois,
                num_splits=1,
                fwhm_mm=fwhm_mm,
                resolution_mm=resolution_mm,
            )
            rows.append(
                {
                    "group": group,
                    "label": label,
                    "ckpt": ckpt,
                    "metric": "localizer_decode_r",
                    "metric_label": "Localizer decoding R",
                    "value": float(
                        np.mean(
                            [
                                decode_scores[
                                    :,
                                    decode_run_rois.index(roi),
                                    decode_run_rois.index(roi),
                                ].mean()
                                for roi in decode_rois
                            ]
                        )
                    ),
                }
            )

            smoothness_result = smoothness(
                ckpt,
                "pitcher",
                fwhm_mm=fwhm_mm,
                resolution_mm=resolution_mm,
            )
            rows.append(
                {
                    "group": group,
                    "label": label,
                    "ckpt": ckpt,
                    "metric": "pitcher_moving_morans_i",
                    "metric_label": "Pitcher moving Moran's I",
                    "value": _moving_mean_from_smoothness(smoothness_result),
                }
            )

        if group in {"LLCNN", "TOPONETS"}:
            continue
        try:
            motion_tvals = collect_localizer_tvals(
                ckpts,
                dataset="robert",
                ret_merged=True,
                verbose=True,
            )
            motion_mae = plot_motion_rois(
                motion_tvals,
                ckpts,
                REBUTTAL_MOTION_ROIS,
                store_dir=None,
            )
            for ckpt, value in zip(ckpts, motion_mae):
                rows.append(
                    {
                        "group": group,
                        "label": label,
                        "ckpt": ckpt,
                        "metric": "robert_motion_mae",
                        "metric_label": "Robert motion MAE",
                        "value": float(value),
                    }
                )
        except Exception as exc:
            print(f"[WARN] Skipping Robert motion MAE for {label}: {exc}")

    return pd.DataFrame(rows)


def collect_base_topo_region_points(
    fwhm_mm=2.0,
    resolution_mm=1.0,
    max_clip_seeds=None,
    add_models=None,
    only_groups=None,
):
    """Per-region values for the base-vs-TopoTransform rebuttal plot."""
    rows = []
    method_groups = _existing_rebuttal_groups(max_clip_seeds=max_clip_seeds, add_models=add_models)
    method_groups = {
        group: ckpts
        for group, ckpts in method_groups.items()
        if group in REBUTTAL_BASE_TOPO_GROUPS
    }
    if only_groups is not None:
        only_groups = set(only_groups)
        method_groups = {
            group: ckpts
            for group, ckpts in method_groups.items()
            if group in only_groups
        }

    for group in REBUTTAL_BASE_TOPO_GROUPS:
        ckpts = method_groups.get(group, [])
        if not ckpts:
            continue
        label = REBUTTAL_METHOD_LABELS.get(group, METHOD_LABELS.get(group, group))
        decode_rois = _decode_rois_for_group(group)
        ceilings = localizer_decode_ceiling(decode_rois, folds=10)
        ceiling_by_roi = {
            roi: float(ceilings[idx].mean(-1).mean())
            for idx, roi in enumerate(decode_rois)
        }

        for ckpt in ckpts:
            decode_run_rois = _decode_run_rois_for_group(group)
            decode_scores = localizer_decode_clustered(
                ckpt,
                decode_run_rois,
                num_splits=1,
                fwhm_mm=fwhm_mm,
                resolution_mm=resolution_mm,
            )
            for roi in decode_rois:
                roi_idx = decode_run_rois.index(roi)
                raw_value = float(decode_scores[:, roi_idx, roi_idx].mean())
                ceiling_value = ceiling_by_roi[roi]
                rows.append(
                    {
                        "group": group,
                        "label": label,
                        "ckpt": ckpt,
                        "metric": "localizer_decode_r",
                        "metric_label": "Normalized localizer decoding R",
                        "region": roi,
                        "region_label": ROI_LABELS[roi],
                        "raw_value": raw_value,
                        "ceiling_value": ceiling_value,
                        "value": raw_value / ceiling_value if ceiling_value > 0 else np.nan,
                    }
                )

            smoothness_result = smoothness(
                ckpt,
                "pitcher",
                fwhm_mm=fwhm_mm,
                resolution_mm=resolution_mm,
            )
            for category, values in smoothness_result.items():
                if "moving" not in category.lower():
                    continue
                rows.append(
                    {
                        "group": group,
                        "label": label,
                        "ckpt": ckpt,
                        "metric": "pitcher_moving_morans_i",
                        "metric_label": "Pitcher moving Moran's I",
                        "region": category,
                        "region_label": category.replace("_moving", "").replace("_", " "),
                        "value": float(values["model_smoothness"]),
                        "human_value": float(values["human_smoothness"]),
                    }
                )

    return pd.DataFrame(rows)


def plot_existing_experiment_metrics(df, out_path, group_order=None, figsize=None):
    metric_order = [
        "localizer_decode_r",
        "pitcher_moving_morans_i",
        "robert_motion_mae",
    ]
    metric_labels = {
        "localizer_decode_r": "Localizer decoding\nmean R",
        "pitcher_moving_morans_i": "Spatial smoothness\nMoran's I",
        "robert_motion_mae": "Motion-localizer\nMAE",
    }
    if group_order is None:
        group_order = REBUTTAL_METHOD_ORDER
    groups = [group for group in group_order if group in set(df["group"])]
    if figsize is None:
        figsize = (8.8, 2.75)
    fig, axes = plt.subplots(1, len(metric_order), figsize=figsize, constrained_layout=True)
    if len(metric_order) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_order):
        sub = df[df["metric"] == metric]
        means = sub.groupby("group")["value"].mean().reindex(groups)
        sems = sub.groupby("group")["value"].sem().reindex(groups)
        x = np.arange(len(groups))
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[REBUTTAL_METHOD_COLORS.get(group, "#BDBDBD") for group in groups],
            capsize=3,
            edgecolor="none",
        )
        for idx, group in enumerate(groups):
            vals = sub[sub["group"] == group]["value"].to_numpy()
            ax.scatter(
                np.full(vals.shape, idx),
                vals,
                color="black",
                s=10,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [REBUTTAL_METHOD_LABELS.get(group, METHOD_LABELS.get(group, group)) for group in groups],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("")
        ax.set_title(metric_labels[metric], fontsize=9)
        _style_axis(ax)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_base_topo_region_points(region_df, out_path, seed_df):
    metric_order = [
        "localizer_decode_r",
        "pitcher_moving_morans_i",
    ]
    metric_labels = {
        "localizer_decode_r": "Localizer decoding\nnormalized R",
        "pitcher_moving_morans_i": "Spatial smoothness\nMoran's I",
    }
    groups = [group for group in REBUTTAL_BASE_TOPO_GROUPS if group in set(seed_df["group"])]
    fig, axes = plt.subplots(1, len(metric_order), figsize=(5.35, 2.75), constrained_layout=True)
    if len(metric_order) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_order):
        if metric == "localizer_decode_r":
            sub = region_df[region_df["metric"] == metric]
            plot_values = (
                sub.groupby(["group", "region", "region_label"])["value"]
                .mean()
                .reset_index()
            )
        else:
            plot_values = seed_df[
                seed_df["metric"].eq(metric) & seed_df["group"].isin(groups)
            ].copy()
            plot_values["region"] = plot_values["ckpt"]
            plot_values["region_label"] = plot_values["ckpt"]

        means = plot_values.groupby("group")["value"].mean().reindex(groups)
        sems = plot_values.groupby("group")["value"].sem().reindex(groups)
        x = np.arange(len(groups))
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[REBUTTAL_METHOD_COLORS.get(group, "#BDBDBD") for group in groups],
            capsize=3,
            edgecolor="none",
        )
        for idx, group in enumerate(groups):
            group_values = plot_values[plot_values["group"] == group]
            if group_values.empty:
                continue
            if metric == "localizer_decode_r":
                colors = [
                    all_roi_colors.get(region, ("", "black"))[1]
                    for region in group_values["region"]
                ]
            else:
                colors = ["black"] * len(group_values)
            ax.scatter(
                np.full(len(group_values), idx),
                group_values["value"].to_numpy(),
                color=colors,
                s=14 if metric == "localizer_decode_r" else 8,
                edgecolor="white" if metric == "localizer_decode_r" else "none",
                linewidth=0.25 if metric == "localizer_decode_r" else 0,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [REBUTTAL_METHOD_LABELS.get(group, METHOD_LABELS.get(group, group)) for group in groups],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("")
        ax.set_title(metric_labels[metric], fontsize=9)
        if metric == "pitcher_moving_morans_i" and "human_value" in region_df:
            human_vals = region_df[
                region_df["metric"].eq("pitcher_moving_morans_i")
            ]["human_value"].dropna().to_numpy()
            if human_vals.size:
                human_value = float(np.mean(human_vals))
                ax.axhline(
                    human_value,
                    color=HUMAN_C,
                    linestyle=(0, (3, 2)),
                    linewidth=1.3,
                    zorder=5,
                )
                human_label_x = min(2, len(groups) - 1)
                ax.text(
                    human_label_x,
                    human_value + 0.008,
                    "Human",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=HUMAN_C,
                )
        _style_axis(ax)

    decode_regions = [
        roi
        for roi in REBUTTAL_DECODE_ROIS
        if roi in set(region_df[region_df["metric"].eq("localizer_decode_r")]["region"])
    ]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=all_roi_colors[roi][1],
            markeredgecolor="white",
            markersize=6,
        )
        for roi in decode_regions
    ]
    axes[0].legend(
        handles,
        [ROI_LABELS[roi] for roi in decode_regions],
        frameon=False,
        loc="center right",
        bbox_to_anchor=(-0.28, 0.5),
        borderaxespad=0,
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _add_models_suffix(add_models=None):
    add_models = _normalize_add_models(add_models)
    if not add_models:
        return ""
    return "_add_" + "_".join(sorted(add_models))


def plot_rebuttal_metric_variants(model_df, out_dir, base_topo_region_df=None, add_models=None):
    suffix = _add_models_suffix(add_models)
    merged_plot = out_dir / f"rebuttal_model_clustering{suffix}.svg"
    vjepa_plot = out_dir / f"rebuttal_model_clustering_vjepa_only{suffix}.svg"
    base_topo_plot = out_dir / f"rebuttal_model_clustering_base_vs_topo{suffix}.svg"
    plot_existing_experiment_metrics(model_df, merged_plot)
    plot_existing_experiment_metrics(
        model_df,
        vjepa_plot,
        group_order=REBUTTAL_VJEPA_ONLY_GROUPS,
        figsize=(5.9, 2.75),
    )
    if base_topo_region_df is None:
        base_topo_region_df = model_df[
            model_df["group"].isin(REBUTTAL_BASE_TOPO_GROUPS)
            & model_df["metric"].isin(["localizer_decode_r", "pitcher_moving_morans_i"])
        ].copy()
        base_topo_region_df["region_label"] = base_topo_region_df["ckpt"]
    plot_base_topo_region_points(base_topo_region_df, base_topo_plot, model_df)
    return [merged_plot, vjepa_plot, base_topo_plot]


def _collect_reliability_pair_rows(
    label,
    ckpt_a,
    loaded_a,
    ckpt_b,
    loaded_b,
    rois=HALF_RELIABILITY_ROIS,
):
    t_a, p_a, pos_a = loaded_a
    t_b, p_b, pos_b = loaded_b
    positions_a = _roi_positions(pos_a)
    positions_b = _roi_positions(pos_b)
    if positions_a.shape != positions_b.shape or not np.allclose(positions_a, positions_b):
        print(f"[WARN] Position mismatch for {label}; centroid distances use each model's own positions.")

    rows = []
    for roi in rois:
        ta, pa = _roi_values(roi, t_a, p_a)
        tb, pb = _roi_values(roi, t_b, p_b)
        mask_a = _roi_mask(roi, ta, pa)
        mask_b = _roi_mask(roi, tb, pb)
        centroid_a = _mask_centroid(mask_a, positions_a)
        centroid_b = _mask_centroid(mask_b, positions_b)
        rows.append(
            {
                "comparison": label,
                "ckpt_a": ckpt_a,
                "ckpt_b": ckpt_b,
                "roi": roi,
                "roi_label": ROI_LABELS[roi],
                "dice": _dice(mask_a, mask_b),
                "n_units_a": int(np.sum(mask_a)),
                "n_units_b": int(np.sum(mask_b)),
                "centroid_distance_mm": float(np.linalg.norm(centroid_a - centroid_b)),
            }
        )
    return rows


def collect_half_reliability(device, fwhm_mm, resolution_mm):
    loaded = {}
    for name, ckpt in HALF_CKPTS.items():
        if not _exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint for {name}: {ckpt}")
        print(f"[half] {name}: {ckpt}")
        loaded[name] = _load_localizers(
            ckpt,
            device,
            fwhm_mm,
            resolution_mm,
            dataset_names=HALF_RELIABILITY_DATASETS,
        )

    rows = []
    rows.extend(
        _collect_reliability_pair_rows(
            "half_split_same_seed",
            HALF_CKPTS["half A"],
            loaded["half A"],
            HALF_CKPTS["half B"],
            loaded["half B"],
        )
    )

    full_loaded = {}
    full_ckpts = [ckpt for ckpt in CKPT_GROUPS.get("TopoTransform", []) if _ckpt_available(ckpt)]
    for ckpt in full_ckpts:
        print(f"[seed baseline] {ckpt}")
        full_loaded[ckpt] = _load_localizers(
            ckpt,
            device,
            fwhm_mm,
            resolution_mm,
            dataset_names=HALF_RELIABILITY_DATASETS,
        )
    for i, ckpt_a in enumerate(full_ckpts):
        for ckpt_b in full_ckpts[i + 1:]:
            rows.extend(
                _collect_reliability_pair_rows(
                    "full_dataset_across_seeds",
                    ckpt_a,
                    full_loaded[ckpt_a],
                    ckpt_b,
                    full_loaded[ckpt_b],
                )
            )
    return pd.DataFrame(rows)


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_model_clustering(df, out_path):
    groups = [group for group in [
        "VJEPA raw",
        "VJEPA + TopoTransform",
        "CLIP raw",
        "CLIP + TopoTransform",
        "SOM",
        "VideoMAE raw",
        "VideoMAE + TopoTransform",
    ] if group in set(df["group"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9), width_ratios=[1.0, 1.0])
    x = np.arange(len(groups))
    roi_colors = [all_roi_colors[roi][1] for roi in ROI_ORDER]
    group_colors = {
        "VJEPA raw": "#BDBDBD",
        "VJEPA + TopoTransform": "#7C8DB0",
        "CLIP raw": "#BDBDBD",
        "CLIP + TopoTransform": "#1B9E77",
        "SOM": "#D95F02",
        "VideoMAE raw": "#BDBDBD",
        "VideoMAE + TopoTransform": "#7570B3",
    }

    metrics = [
        ("peak_t", "Functional selectivity\n(peak t)"),
        ("largest_cluster_fraction", "Spatial clustering\n(largest-cluster fraction)"),
    ]
    tick_labels = {
        "VJEPA raw": "VJEPA\nraw",
        "VJEPA + TopoTransform": "VJEPA\n+Topo",
        "CLIP raw": "CLIP\nraw",
        "CLIP + TopoTransform": "CLIP\n+Topo",
        "SOM": "VJEPA-SOM",
        "VideoMAE raw": "VideoMAE\nraw",
        "VideoMAE + TopoTransform": "VideoMAE\n+Topo",
    }
    for ax, (metric, ylabel) in zip(axes, metrics):
        means = df.groupby("group")[metric].mean().reindex(groups)
        sems = df.groupby("group")[metric].sem().reindex(groups)
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[group_colors[group] for group in groups],
            capsize=3,
        )
        for idx, group in enumerate(groups):
            sub = df[df["group"] == group]
            for offset, roi in zip(np.linspace(-0.18, 0.18, len(ROI_ORDER)), ROI_ORDER):
                vals = sub[sub["roi"] == roi][metric].to_numpy()
                ax.scatter(
                    np.full(vals.shape, idx + offset),
                    vals,
                    color=all_roi_colors[roi][1],
                    s=18,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([tick_labels[group] for group in groups])
        ax.set_ylabel(ylabel)
        if metric == "largest_cluster_fraction":
            ax.set_ylim(0, 1.02)
        _style_axis(ax)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=6)
        for color in roi_colors
    ]
    axes[0].legend(
        handles,
        [ROI_LABELS[roi] for roi in ROI_ORDER],
        frameon=False,
        ncol=4,
        loc="upper left",
        bbox_to_anchor=(0, 1.24),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_half_reliability(df, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(4.7, 2.9))
    rois = [roi for roi in HALF_RELIABILITY_ROIS if roi in set(df["roi"])]
    labels = [ROI_LABELS[roi] for roi in rois]
    comparisons = [
        "half_split_same_seed",
        "full_dataset_across_seeds",
    ]
    comparison_labels = {
        "half_split_same_seed": "Half A vs half B",
        "full_dataset_across_seeds": "Across seeds",
    }
    comparison_colors = {
        "half_split_same_seed": "#1B9E77",
        "full_dataset_across_seeds": "#8E8E8E",
    }
    x = np.arange(len(rois))
    width = 0.34

    for comp_idx, comparison in enumerate(comparisons):
        sub = df[df["comparison"] == comparison]
        means = sub.groupby("roi")["dice"].mean().reindex(rois)
        sems = sub.groupby("roi")["dice"].sem().reindex(rois)
        offset = (comp_idx - 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sems,
            color=comparison_colors[comparison],
            capsize=3,
            edgecolor="none",
            label=comparison_labels[comparison],
        )
        for roi_idx, roi in enumerate(rois):
            vals = sub[sub["roi"] == roi]["dice"].to_numpy()
            ax.scatter(
                np.full(vals.shape, x[roi_idx] + offset),
                vals,
                color="black",
                s=8,
                zorder=3,
            )

    significance_y = []
    for roi_idx, roi in enumerate(rois):
        half_vals = df[
            df["comparison"].eq("half_split_same_seed") & df["roi"].eq(roi)
        ]["dice"].to_numpy()
        baseline_vals = df[
            df["comparison"].eq("full_dataset_across_seeds") & df["roi"].eq(roi)
        ]["dice"].to_numpy()
        if half_vals.size == 0 or baseline_vals.size < 2:
            continue
        half_val = float(half_vals[0])
        _, p_val = stats.ttest_1samp(
            baseline_vals,
            popmean=half_val,
            alternative="less",
            nan_policy="omit",
        )
        if np.isfinite(p_val) and p_val < 0.05:
            y = max(half_val, float(np.nanmax(baseline_vals))) + 0.045
            significance_y.append(y)
            ax.text(
                roi_idx,
                y,
                "*",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    data_max = float(np.nanmax(df["dice"].to_numpy()))
    y_max = max([data_max] + significance_y)
    ax.set_ylim(0, min(0.72, y_max + 0.06))
    ax.set_ylabel("Dice overlap")
    ax.set_title("ROI overlap", loc="left", fontweight="bold")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_compact_floc_map(ckpt_name, label, out_dir, device, fwhm_mm, resolution_mm):
    print(f"[map] {label}: {ckpt_name}")
    t_vals_dict, p_vals_dict, layer_positions = _load_localizers(
        ckpt_name, device, fwhm_mm, resolution_mm
    )
    store_dir = ensure_dir(out_dir / label)
    t_threshold = 2.0 if label == "som" else LOCALIZER_T_THRESHOLD
    p_threshold = 0.05 if label == "som" else LOCALIZER_P_THRESHOLD
    plot_all_rois(
        t_vals_dict,
        p_vals_dict,
        layer_positions,
        store_dir,
        figsize_per_panel=4,
        p_threshold=p_threshold,
        t_threshold=t_threshold,
    )


def plot_reliability_and_som_maps(out_dir, device, fwhm_mm, resolution_mm):
    map_dir = ensure_dir(out_dir / "functional_cluster_maps")
    for label, ckpt in HALF_CKPTS.items():
        _plot_compact_floc_map(ckpt, label.replace(" ", "_").lower(), map_dir, device, fwhm_mm, resolution_mm)
    if _exists(SOM_CKPT):
        _plot_compact_floc_map(SOM_CKPT, "som", map_dir, device, fwhm_mm, resolution_mm)


def write_manifest(out_dir):
    rows = []
    for family, ckpts in [
        ("vjepa", [VJEPA_CKPT]),
        ("clip", CLIP_CKPTS),
        ("videomae", VIDEOMAE_CKPTS),
        ("som", [SOM_CKPT]),
        ("llcnn", [LLCNN_CKPT]),
        ("toponets", [TOPONETS_CKPT]),
        ("half", list(HALF_CKPTS.values())),
        ("small", [SMALL_CKPT]),
    ]:
        for ckpt in ckpts:
            rows.append({"family": family, "ckpt": ckpt, "exists": _ckpt_available(ckpt)})
    df = pd.DataFrame(rows)
    path = out_dir / "rebuttal_checkpoint_manifest.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"Saved manifest: {path}")
    return df


def _replace_groups(df, new_df):
    if df is None or df.empty:
        return new_df
    if new_df is None or new_df.empty:
        return df
    groups = set(new_df["group"])
    return pd.concat([df[~df["group"].isin(groups)], new_df], ignore_index=True)


def _filter_add_model_display(df, add_models=None):
    if df is None or df.empty or not _normalize_add_models(add_models):
        return df
    return df[~df["group"].isin(REBUTTAL_ADD_MODEL_DROP_GROUPS)].copy()


def main():
    parser = argparse.ArgumentParser(description="Concise rebuttal plots for model generality and reliability.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fwhm-mm", type=float, default=2.0)
    parser.add_argument("--resolution-mm", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(PLOTS_DIR / "rebuttal"))
    parser.add_argument("--include-videomae", action="store_true")
    parser.add_argument("--max-clip-seeds", type=int, default=None)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-half", action="store_true")
    parser.add_argument("--skip-maps", action="store_true")
    parser.add_argument(
        "--add-models",
        nargs="*",
        default=None,
        help="Optional extra model baselines to include. With no names, adds llcnn, toponets, and swapopt.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    write_manifest(out_dir)

    if not args.skip_models:
        model_csv = out_dir / "rebuttal_model_clustering.csv"
        base_topo_region_csv = out_dir / "rebuttal_model_clustering_base_vs_topo_region_points.csv"
        add_models = _normalize_add_models(args.add_models)
        if add_models and model_csv.exists() and base_topo_region_csv.exists():
            model_df = pd.read_csv(model_csv)
            base_topo_region_df = pd.read_csv(base_topo_region_csv)
            add_groups = set()
            if "llcnn" in add_models:
                add_groups.add("LLCNN")
            if "toponets" in add_models:
                add_groups.add("TOPONETS")
            if "swapopt" in add_models:
                add_groups.add("SWAPOPT")
            if "tdann" in add_models:
                add_groups.add("TDANN")
            if add_groups:
                new_model_df = collect_existing_experiment_metrics(
                    args.fwhm_mm,
                    args.resolution_mm,
                    max_clip_seeds=args.max_clip_seeds,
                    add_models=args.add_models,
                    only_groups=add_groups,
                )
                new_base_topo_region_df = collect_base_topo_region_points(
                    args.fwhm_mm,
                    args.resolution_mm,
                    max_clip_seeds=args.max_clip_seeds,
                    add_models=args.add_models,
                    only_groups=add_groups,
                )
                model_df = _replace_groups(model_df, new_model_df)
                base_topo_region_df = _replace_groups(base_topo_region_df, new_base_topo_region_df)
        else:
            model_df = collect_existing_experiment_metrics(
                args.fwhm_mm,
                args.resolution_mm,
                max_clip_seeds=args.max_clip_seeds,
                add_models=args.add_models,
            )
            base_topo_region_df = collect_base_topo_region_points(
                args.fwhm_mm,
                args.resolution_mm,
                max_clip_seeds=args.max_clip_seeds,
                add_models=args.add_models,
            )
        model_df = _filter_add_model_display(model_df, args.add_models)
        base_topo_region_df = _filter_add_model_display(base_topo_region_df, args.add_models)
        base_topo_region_df = base_topo_region_df[
            base_topo_region_df["group"].isin(REBUTTAL_BASE_TOPO_GROUPS)
        ].copy()
        model_df.to_csv(model_csv, index=False)
        base_topo_region_df.to_csv(base_topo_region_csv, index=False)
        model_plots = plot_rebuttal_metric_variants(
            model_df,
            out_dir,
            base_topo_region_df,
            add_models=args.add_models,
        )
        print(f"Saved existing-experiment summary CSV: {model_csv}")
        print(f"Saved base-vs-topo region-point CSV: {base_topo_region_csv}")
        for model_plot in model_plots:
            print(f"Saved existing-experiment summary plot: {model_plot}")
        summary = model_df.groupby(["label", "metric"])["value"].agg(["mean", "sem", "count"])
        print(summary.to_string())

    if not args.skip_half:
        half_df = collect_half_reliability(args.device, args.fwhm_mm, args.resolution_mm)
        half_csv = out_dir / "rebuttal_half_split_reliability.csv"
        half_plot = out_dir / "rebuttal_half_split_reliability.svg"
        half_df.to_csv(half_csv, index=False)
        plot_half_reliability(half_df, half_plot)
        print(f"Saved half reliability CSV: {half_csv}")
        print(f"Saved half reliability plot: {half_plot}")
        print(half_df.to_string(index=False))
        for comparison, sub in half_df.groupby("comparison"):
            print(
                f"{comparison} mean Dice: "
                f"{np.nanmean(sub['dice']):.3f} +/- {stats.sem(sub['dice'], nan_policy='omit'):.3f}"
            )

    if not args.skip_maps:
        plot_reliability_and_som_maps(
            out_dir,
            args.device,
            args.fwhm_mm,
            args.resolution_mm,
        )


if __name__ == "__main__":
    main()
