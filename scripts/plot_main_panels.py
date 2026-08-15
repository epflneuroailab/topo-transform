"""Render publication-specific main-text graph panels and localizer maps."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import PLOTS_DIR
from .analysis_utils import (
    CKPT_GROUPS,
    METHOD_COLORS,
    METHOD_LABELS,
    collect_localizer_tvals,
    resolve_group_names,
)
from .common import MODEL_CKPT
from .get_localizers import localizers
from .plot_localizer_decode import plot_localizer_decode
from .plot_localizer_motion import plot_all_rois as plot_motion_rois
from .plot_localizers import plot_all_rois as plot_localizer_maps
from .plot_smoothness import DEFAULT_FWHM, _collect_group_moving, _draw_label
from .plot_utils import ensure_dir, savefig


MAIN_METHODS = ("TopoTransform", "TDANN", "SWAPOPT", "UNOPTIMIZED")
MOTION_ROIS = ("face", "body", "place", "mt", "v6", "psts")
SMOOTHNESS_GROUPS = (
    ("UNOPTIMIZED", "no smoothing", 0.0),
    ("UNOPTIMIZED", None, DEFAULT_FWHM),
    ("SWAPOPT", None, DEFAULT_FWHM),
    ("TDANN", None, DEFAULT_FWHM),
    ("TopoTransform", "no smoothing", 0.0),
    ("TopoTransform", None, DEFAULT_FWHM),
)


def render_localizer_maps():
    groups = {
        "topotransform": MODEL_CKPT,
        "tdann": CKPT_GROUPS["TDANN"][0],
        "swapopt": CKPT_GROUPS["SWAPOPT"][0],
    }
    for output_name, checkpoint in groups.items():
        t_values, p_values, positions = localizers(checkpoint, ret_merged=True)
        plot_localizer_maps(
            t_values,
            p_values,
            positions,
            PLOTS_DIR / "localizers" / output_name,
        )


def render_localizer_decode():
    methods = resolve_group_names(MAIN_METHODS)
    results = {
        name: plot_localizer_decode(CKPT_GROUPS[name], store_dir=None)
        for name in methods
    }
    human_scores = results[methods[0]][1].mean(axis=0)
    model_scores = [results[name][0].mean(axis=0) for name in methods]
    labels = [METHOD_LABELS[name] for name in methods]
    colors = [METHOD_COLORS[name] for name in methods]

    plt.figure(figsize=(2.2, 2.0))
    bars = plt.barh(
        labels,
        [np.mean(scores) for scores in model_scores],
        color=colors,
        height=0.71,
    )
    for bar, label in zip(bars, labels):
        compact = label == "SwapOpt"
        plt.text(
            0.005 if compact else 0.03,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            color="white",
            fontsize=6 if compact else 10,
        )
    for index, scores in enumerate(model_scores):
        plt.scatter(scores, np.full(len(scores), index), color="black", s=5)

    ci = 1.96 * human_scores.std()
    plt.fill_betweenx(
        [-1, len(labels)],
        human_scores.mean() - ci,
        human_scores.mean() + ci,
        color="#2E7D32",
        alpha=0.3,
        edgecolor="none",
    )
    plt.xlabel("Mean prediction score (R)")
    plt.yticks([])
    plt.ylim(-0.8, len(labels) - 0.2)
    plt.xlim(0, 0.6)
    sns.despine()
    savefig(
        PLOTS_DIR / "localizer_decoding_score_comparison.svg",
        dpi=300,
        bbox_inches="tight",
    )


def render_smoothness():
    values = []
    human_reference = None
    for group, sublabel, fwhm in SMOOTHNESS_GROUPS:
        model_values, human_value = _collect_group_moving(CKPT_GROUPS[group], fwhm)
        values.append((group, sublabel, model_values))
        if human_reference is None:
            human_reference = human_value

    positions = np.arange(len(values))
    fig, axes = plt.subplots(figsize=(2.6, 2.6))
    colors = [METHOD_COLORS.get(group, "#808080") for group, _, _ in values]
    axes.barh(
        positions,
        [float(np.mean(group_values)) for _, _, group_values in values],
        height=0.75,
        color=colors,
        edgecolor="none",
    )
    for index, (_, _, group_values) in enumerate(values):
        axes.scatter(
            group_values,
            np.full(len(group_values), index),
            color="black",
            s=8,
            zorder=10,
        )
    for index, (group, sublabel, _) in enumerate(values):
        _draw_label(
            axes,
            0.02,
            positions[index],
            METHOD_LABELS.get(group, group),
            sub=f"({sublabel})" if sublabel else None,
        )

    axes.axvline(
        human_reference,
        color="#2E7D32",
        linestyle="--",
        linewidth=3.0,
        zorder=5,
    )
    axes.set_xlim(0, 1)
    axes.set_yticks([])
    axes.invert_yaxis()
    axes.set_xlabel("Spatial autocorrelation (Moran's I)", fontsize=9)
    axes.tick_params(axis="both", which="major", labelsize=8)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig(
        PLOTS_DIR / "smoothness_comparison_bar.svg",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


def render_motion_selectivity():
    store_dir = ensure_dir(PLOTS_DIR / "localizer_motion")
    methods = resolve_group_names(MAIN_METHODS)
    mae_by_method = {}
    for name in methods:
        checkpoints = CKPT_GROUPS[name]
        all_t_values = collect_localizer_tvals(
            checkpoints,
            dataset="robert",
            ret_merged=True,
            verbose=False,
        )
        mae_by_method[name] = plot_motion_rois(
            all_t_values,
            checkpoints,
            list(MOTION_ROIS),
            ensure_dir(store_dir / name.lower()),
        )

    maes = [mae_by_method[name] for name in methods]
    labels = [
        "Ours" if name == "TopoTransform" else METHOD_LABELS[name]
        for name in methods
    ]
    colors = [METHOD_COLORS[name] for name in methods]
    plt.figure(figsize=(3.3, 2.7))
    y_positions = np.arange(len(methods))
    bars = plt.barh(y_positions, [values.mean() for values in maes], color=colors)
    for index, values in enumerate(maes):
        plt.plot(values, [index] * len(values), "ko", markersize=4)
    for bar, label in zip(bars, labels):
        plt.text(
            0.03,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            color="white",
            fontsize=10,
        )
    axes = plt.gca()
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    plt.yticks([])
    axes.tick_params(axis="y", length=0)
    plt.xlabel("Mean absolute error", fontsize=12)
    plt.tight_layout()
    savefig(store_dir / "localizer_motion_mae_comparison.svg")


def main():
    render_localizer_maps()
    render_localizer_decode()
    render_smoothness()
    render_motion_selectivity()


if __name__ == "__main__":
    main()
