"""Plot VJEPA layer-to-ROI neural alignment.

This script reads the existing brainscore_analysis layer-decoding cache:
`cache/test_vjepa_layers_decoding/full.pkl.p`.

The score cache was produced on voxels with joint ceiling > 0.4. We reconstruct
the same valid voxel axis from cached per-dataset ceilings, then intersect it
with Glasser V1/V4/FFC masks.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from neuroparc import atlas as natlas

from config import PLOTS_DIR
from scripts.get_localizers import get_localizer_human


BRAINSCORE_CACHE = Path("/mnt/scratch/ytang/brainscore_analysis/cache")
DEFAULT_DATASETS = (
    "McMahon2023-fMRI",
    "Lahner2024-fMRI",
    "Keles2024-fMRI",
    "Berezutskaya2021-fMRI",
    "Savasegal2023-fMRI-Defeat",
    "Savasegal2023-fMRI-Growth",
    "Savasegal2023-fMRI-Iteration",
    "Savasegal2023-fMRI-Lemonade",
)
ROI_TO_GLASSER_NAME = {
    "V1": "Primary_Visual_Cortex",
    "V4": "Fourth_Visual_Area",
    "FFC": "Fusiform_Face_Complex",
    "FFA": "Fusiform_Face_Complex",
}
LOCALIZER_ROIS = {
    "face": "face",
    "Face": "face",
}
ROI_COLORS = {
    "V1": "#2E7D32",
    "V4": "#E69F00",
    "FFC": "#C73E1D",
    "FFA": "#C73E1D",
    "face": "#C73E1D",
    "Face": "#C73E1D",
}
ROI_LABELS = {
    "FFC": "FFA",
    "face": "Face ROI",
    "Face": "Face ROI",
}


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_joint_ceiling(cache_dir: Path, datasets: tuple[str, ...]) -> np.ndarray:
    weighted = 0.0
    total_weight = 0.0
    for dataset in datasets:
        ceiling = _load_pickle(cache_dir / "ceiling" / f"{dataset}.p")
        meta = _load_pickle(cache_dir / "meta" / f"{dataset}.p")
        weight = (
            meta["time_bin_duration"]
            * meta["num_time_bins"]
            * meta["num_presentations"]
        )
        weighted = weighted + ceiling * weight
        total_weight += weight
    return weighted / total_weight


def load_scores(cache_dir: Path, mode: str) -> np.ndarray:
    scores = _load_pickle(
        cache_dir / "test_vjepa_layers_decoding" / f"{mode}.pkl.p"
    )
    if scores.ndim != 4 or scores.shape[1] != 1 or scores.shape[2] < 2:
        raise ValueError(f"Unexpected score shape: {scores.shape}")
    return scores[:, 0, 1, :]


def get_glasser_masks(rois: tuple[str, ...]) -> dict[str, np.ndarray]:
    atlas = natlas.Atlas("Glasser")
    labels = atlas.label_surface("fsaverage5")
    label_lookup = {k.lower(): v for k, v in atlas.rev_label_name_map.items()}
    masks = {}
    for roi in rois:
        if roi in LOCALIZER_ROIS:
            masks[roi] = get_localizer_human([LOCALIZER_ROIS[roi]])[0]
            continue
        glasser_name = ROI_TO_GLASSER_NAME.get(roi, roi)
        label = label_lookup[glasser_name.lower()]
        masks[roi] = np.isin(labels, [label])
    return masks


def compute_roi_curves(
    scores: np.ndarray,
    ceiling: np.ndarray,
    roi_masks: dict[str, np.ndarray],
    ceiling_threshold: float,
    metric: str,
    score_valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    ceiling_mean = ceiling.mean(axis=0)
    if score_valid_mask is None:
        valid = ceiling_mean > ceiling_threshold
    else:
        valid = np.asarray(score_valid_mask, dtype=bool)
        if valid.shape != ceiling_mean.shape:
            raise ValueError(
                "Score valid mask and ceiling have incompatible shapes: "
                f"{valid.shape} vs {ceiling_mean.shape}"
            )

    if scores.shape[-1] != int(valid.sum()):
        raise ValueError(
            "Score voxel axis does not match ceiling-thresholded voxels: "
            f"{scores.shape[-1]} scores vs {int(valid.sum())} valid voxels"
        )

    valid_ceiling = ceiling_mean[valid]
    selected_valid = ceiling_mean[valid] > ceiling_threshold
    curves = {}
    for roi, mask in roi_masks.items():
        roi_valid = mask[valid] & selected_valid
        if not np.any(roi_valid):
            raise ValueError(f"ROI {roi!r} has no voxels after ceiling threshold")

        roi_scores = scores[:, roi_valid]
        if metric == "r":
            curve = roi_scores.mean(axis=1)
        elif metric == "r_over_ceiling":
            denom = np.maximum(valid_ceiling[roi_valid], 1e-8)
            curve = (roi_scores / denom).mean(axis=1)
        elif metric == "r2":
            curve = np.square(roi_scores).mean(axis=1)
        elif metric == "r2_over_ceiling":
            denom = np.maximum(valid_ceiling[roi_valid], 1e-8)
            curve = (np.square(roi_scores) / denom).mean(axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        curves[roi] = curve
    return curves


def plot_curves(
    curves: dict[str, np.ndarray],
    metric: str,
    output: Path,
    title: str,
) -> None:
    layers = np.arange(next(iter(curves.values())).shape[0])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    for roi, curve in curves.items():
        label = ROI_LABELS.get(roi, roi)
        best_layer = int(np.nanargmax(curve))
        best_value = float(curve[best_layer])
        ax.plot(
            layers,
            curve,
            marker="o",
            markersize=3.5,
            linewidth=2,
            label=label,
            color=ROI_COLORS.get(roi),
        )
        ax.scatter(
            [best_layer],
            [best_value],
            s=95,
            facecolors="none",
            edgecolors="#D62728",
            linewidths=1.8,
            zorder=5,
        )
        text_offset = (0, -18) if label in {"V4", "FFA"} else (0, 12)
        va = "top" if label in {"V4", "FFA"} else "bottom"
        ax.annotate(
            f"L{best_layer}",
            xy=(best_layer, best_value),
            xytext=text_offset,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color="black",
            arrowprops={
                "arrowstyle": "->",
                "color": "black",
                "linewidth": 0.9,
                "shrinkA": 0,
                "shrinkB": 5,
            },
        )

    ylabel = {
        "r": "Pearson r",
        "r_over_ceiling": "Normalized predictivity (R / ceiling)",
        "r2": "Variance explained (r^2)",
        "r2_over_ceiling": "Noise-corrected variance explained",
    }[metric]
    ax.set_xlabel("VJEPA layer (early to late)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(0, len(layers), 2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output}")


def plot_best_layer_bars(
    curves: dict[str, np.ndarray],
    metric: str,
    output: Path,
    title: str,
) -> None:
    rois = list(curves)
    labels = [ROI_LABELS.get(roi, roi) for roi in rois]
    values = np.array([float(np.nanmax(curves[roi])) for roi in rois])
    best_layers = [int(np.nanargmax(curves[roi])) for roi in rois]
    colors = [ROI_COLORS.get(roi, "#555555") for roi in rois]

    ylabel = {
        "r": "Peak Pearson r",
        "r_over_ceiling": "Peak normalized predictivity (R / ceiling)",
        "r2": "Peak variance explained (r^2)",
        "r2_over_ceiling": "Peak noise-corrected variance explained",
    }[metric]

    fig, ax = plt.subplots(figsize=(3.2, 3.1))
    x = np.arange(len(rois))
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="none")
    for bar, value, layer in zip(bars, values, best_layers):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values.max() * 0.025, 0.01),
            f"L{layer}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, values.max() * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved bar plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=BRAINSCORE_CACHE)
    parser.add_argument("--mode", choices=("full", "sampled"), default="full")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--rois", nargs="+", default=["V1", "V4", "FFC"])
    parser.add_argument("--ceiling_threshold", type=float, default=0.4)
    parser.add_argument(
        "--metric",
        choices=("r", "r_over_ceiling", "r2", "r2_over_ceiling"),
        default="r",
    )
    parser.add_argument("--plot-type", choices=("curves", "bars"), default="curves")
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS_DIR / "vjepa_layer_roi_pearson_r.svg",
    )
    args = parser.parse_args()

    rois = tuple(args.rois)
    ceiling = load_joint_ceiling(args.cache_dir, tuple(args.datasets))
    score_ceiling = load_joint_ceiling(args.cache_dir, DEFAULT_DATASETS)
    score_valid_mask = score_ceiling.mean(axis=0) > args.ceiling_threshold
    scores = load_scores(args.cache_dir, args.mode)
    roi_masks = get_glasser_masks(rois)
    curves = compute_roi_curves(
        scores=scores,
        ceiling=ceiling,
        roi_masks=roi_masks,
        ceiling_threshold=args.ceiling_threshold,
        metric=args.metric,
        score_valid_mask=score_valid_mask,
    )

    for roi, curve in curves.items():
        best_layer = int(np.nanargmax(curve))
        print(f"{roi}: best layer {best_layer}, peak {curve[best_layer]:.4f}")

    title = "VJEPA layer decoding by Glasser ROI"
    if args.plot_type == "bars":
        plot_best_layer_bars(curves, metric=args.metric, output=args.output, title=title)
    else:
        plot_curves(curves, metric=args.metric, output=args.output, title=title)


if __name__ == "__main__":
    main()
