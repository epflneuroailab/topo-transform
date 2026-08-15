"""Regenerate cache-backed supplementary graph panels with publication labels."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import CACHE_DIR, PLOTS_DIR
from .common import MODEL_CKPTS
from .export_source_data import (
    build_si_S2,
    build_si_S3,
    build_si_S4a,
    build_si_S4c,
    build_si_S8a,
    build_si_S8b,
    build_si_S9,
)
from .get_localizers import localizers
from .plot_localizers import plot_all_rois
from .plot_utils import ensure_dir, savefig
from .plot_vjepa_layer_roi_variance import (
    BRAINSCORE_CACHE,
    DEFAULT_DATASETS,
    compute_roi_curves,
    get_glasser_masks,
    load_joint_ceiling,
    load_scores,
)


OUT_DIR = Path(PLOTS_DIR) / "supplementary"
ROI_COLORS = {"V1": "#1f77b4", "V4": "#2ca02c", "FFC": "#d62728"}
ROI_LABELS = {"V1": "V1", "V4": "V4", "FFC": "FFC"}
SELECTED_LAYERS = {"V1": 15, "V4": 17, "FFC": 23}
S7_SEEDS = (46,)


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_s1():
    path = Path(CACHE_DIR) / "debug" / "figS1_curves.json"
    payload = json.loads(path.read_text())
    blocks = np.asarray(payload["blocks"], dtype=int)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for roi in ("V1", "V4", "IT"):
        ax.plot(blocks, payload["curves"][roi], marker="o", markersize=3, label=roi)
    ax.axvspan(14, 23, color="#d9d9d9", alpha=0.55, linewidth=0)
    ax.set_xlabel("V-JEPA block")
    ax.set_ylabel("Normalized variance explained")
    ax.set_xticks(blocks)
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    savefig(OUT_DIR / "figS1_layer_alignment.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _values(row, start, stop):
    values = []
    for value in row[start:stop]:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return np.asarray(values, dtype=float)


def _sem(values):
    return values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0


def plot_s2_s3():
    for number, builder in ((2, build_si_S2), (3, build_si_S3)):
        payload = builder()
        decode_rows = payload["blocks"][0]["rows"]
        smooth_rows = payload["blocks"][1]["rows"]
        human = float(payload["blocks"][2]["rows"][0][1])
        labels = [row[0] for row in decode_rows]
        x = np.arange(len(labels))

        decode_values = [_values(row, 1, -2) for row in decode_rows]
        smooth_values = [_values(row, 1, -2) for row in smooth_rows]
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

        axes[0].bar(x, [values.mean() for values in decode_values],
                    yerr=[_sem(values) for values in decode_values], color="#7f94bd", capsize=3)
        for index, values in enumerate(decode_values):
            axes[0].scatter(index + np.linspace(-0.12, 0.12, len(values)), values,
                            s=12, color="#333333", zorder=3)
        axes[0].set_ylabel("Pearson R / internal consistency")
        axes[0].set_xticks(x, labels, rotation=30, ha="right")
        axes[0].set_title("Functional alignment")

        axes[1].bar(x, [values.mean() for values in smooth_values],
                    yerr=[_sem(values) for values in smooth_values], color="#8c8c8c", capsize=3)
        for index, values in enumerate(smooth_values):
            axes[1].scatter(index + np.linspace(-0.12, 0.12, len(values)), values,
                            s=12, color="black", zorder=3)
        axes[1].axhline(human, color="#2E7D32", linestyle="--", linewidth=1.5)
        axes[1].set_ylabel("Moran's I")
        axes[1].set_xticks(x, labels, rotation=30, ha="right")
        axes[1].set_title("Spatial smoothness")

        for ax in axes:
            _style_axis(ax)
        fig.tight_layout()
        name = "figS2_controls.svg" if number == 2 else "figS3_static_models.svg"
        savefig(OUT_DIR / "exact" / name, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_model_summary(payload, output, ylabel, human_band=False):
    rows = payload["blocks"][0]["rows"]
    labels = [row[0] for row in rows]
    values = [np.asarray(row[1:-1], dtype=float) for row in rows]
    means = np.asarray([value.mean() for value in values])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.barh(y, means, color=["#7f94bd"] + ["#8c8c8c"] * (len(rows) - 1))
    for index, group in enumerate(values):
        ax.scatter(group, index + np.linspace(-0.10, 0.10, len(group)), s=15,
                   color="black", zorder=3)
    if human_band:
        human = dict(payload["blocks"][1]["rows"])
        ax.axvspan(human["mean"] - human["95% CI half-width"],
                   human["mean"] + human["95% CI half-width"],
                   color="#2E7D32", alpha=0.28, linewidth=0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    _style_axis(ax)
    fig.tight_layout()
    savefig(OUT_DIR / "exact" / output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_s4():
    _plot_model_summary(
        build_si_S4a(),
        "figS4a_alignment.svg",
        "Mean prediction score (R)",
        human_band=True,
    )
    _plot_model_summary(
        build_si_S4c(),
        "figS4c_motion_mae.svg",
        "Mean absolute error",
    )


def plot_s9():
    payload = build_si_S9()
    rows = payload["blocks"][0]["rows"]
    epochs = np.asarray([row[0] for row in rows], dtype=int)
    neural = np.asarray([row[1] for row in rows], dtype=float)
    task = np.asarray([row[2] for row in rows], dtype=float)
    topo = np.asarray([row[3] for row in rows], dtype=float)
    baseline = dict(payload["blocks"][1]["rows"])

    fig, (ax_loss, ax_joint) = plt.subplots(1, 2, figsize=(6.2, 3.0), sharex=True)
    ax_loss.plot(epochs, topo, marker="o", color="#C62828", linewidth=1.6)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Topographic loss")

    ax_joint.plot(epochs, neural, marker="o", color="#000000", linewidth=1.6,
                  label="Neural alignment")
    ax_joint.axhline(baseline["neural alignment (pre)"], color="#000000",
                     linestyle="--", linewidth=1.2, label="Neural alignment (pre)")
    ax_joint.set_xlabel("Epoch")
    ax_joint.set_ylabel("Normalized neural alignment")

    ax_task = ax_joint.twinx()
    ax_task.plot(epochs, task, marker="o", color="#6E6E6E", linewidth=1.6,
                 label="Task performance")
    ax_task.axhline(baseline["task performance (pre)"], color="#6E6E6E",
                    linestyle="--", linewidth=1.2, label="Task performance (pre)")
    ax_task.set_ylabel("Task performance")

    for ax in (ax_loss, ax_joint):
        _style_axis(ax)
    ax_task.spines["top"].set_visible(False)
    handles_a, labels_a = ax_joint.get_legend_handles_labels()
    handles_b, labels_b = ax_task.get_legend_handles_labels()
    ax_joint.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(OUT_DIR / "figS9_training_prepost.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_s8():
    for letter, builder in (("a", build_si_S8a), ("b", build_si_S8b)):
        payload = builder()
        model_rows = payload["blocks"][0]["rows"]
        human_rows = payload["blocks"][1]["rows"]
        rois = [row[0] for row in model_rows]
        model_values = np.asarray([row[1:6] for row in model_rows], dtype=float)
        human_values = np.asarray([row[1:16] for row in human_rows], dtype=float)
        x = np.arange(len(rois))
        width = 0.36

        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        ax.bar(x - width / 2, model_values.mean(axis=1), width,
               yerr=model_values.std(axis=1), color="#7f94bd", capsize=3, label="TopoTransform")
        ax.bar(x + width / 2, human_values.mean(axis=1), width,
               yerr=human_values.std(axis=1), color="#8c8c8c", capsize=3, label="Human")
        for index, values in enumerate(model_values):
            jitter = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(index - width / 2 + jitter, values, s=9, color="black", zorder=3)
        for index, values in enumerate(human_values):
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(index + width / 2 + jitter, values, s=7, color="black", alpha=0.75, zorder=3)
        ax.set_xticks(x, rois)
        ax.set_ylabel("Proportion motion-selective")
        ax.set_ylim(0, 1.08)
        ax.legend(frameon=False, ncol=2)
        _style_axis(ax)
        fig.tight_layout()
        savefig(OUT_DIR / f"figS8{letter}_motion_threshold.svg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_s7():
    base_dir = ensure_dir(PLOTS_DIR / "localizers" / "topotransform" / "seeds")
    for checkpoint in MODEL_CKPTS:
        if not any(f"sd{seed}.pt" in checkpoint for seed in S7_SEEDS):
            continue
        output_dir = ensure_dir(base_dir / Path(checkpoint).stem)
        t_values, p_values, positions = localizers(checkpoint, ret_merged=True)
        plot_all_rois(t_values, p_values, positions, output_dir)


def _roi_curves(metric):
    ceiling = load_joint_ceiling(BRAINSCORE_CACHE, DEFAULT_DATASETS)
    valid = ceiling.mean(axis=0) > 0.4
    scores = load_scores(BRAINSCORE_CACHE, "full")
    masks = get_glasser_masks(tuple(SELECTED_LAYERS))
    return compute_roi_curves(
        scores=scores,
        ceiling=ceiling,
        roi_masks=masks,
        ceiling_threshold=0.4,
        metric=metric,
        score_valid_mask=valid,
    )


def plot_s15_s16():
    raw = _roi_curves("r")
    normalized = _roi_curves("r_over_ceiling")
    layers = np.arange(24)

    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    for roi in SELECTED_LAYERS:
        ax.plot(layers, raw[roi], marker="o", markersize=3,
                color=ROI_COLORS[roi], label=ROI_LABELS[roi])
        layer = SELECTED_LAYERS[roi]
        ax.scatter([layer], [raw[roi][layer]], s=65, facecolors="none",
                   edgecolors="#D62728", linewidths=1.4, zorder=5)
    ax.set_xlabel("V-JEPA layer")
    ax.set_ylabel("Pearson r")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    savefig(OUT_DIR / "figS15_fmri_layer.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rois = list(SELECTED_LAYERS)
    values = np.asarray([normalized[roi][SELECTED_LAYERS[roi]] for roi in rois])
    fig, ax = plt.subplots(figsize=(3.2, 3.1))
    x = np.arange(len(rois))
    bars = ax.bar(x, values, color=[ROI_COLORS[roi] for roi in rois], width=0.62)
    for bar, roi, value in zip(bars, rois, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                f"L{SELECTED_LAYERS[roi]}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [ROI_LABELS[roi] for roi in rois])
    ax.set_ylabel("Pearson R / internal consistency")
    ax.set_ylim(0, max(values) * 1.18)
    _style_axis(ax)
    fig.tight_layout()
    savefig(OUT_DIR / "figS16_normalized_alignment.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for roi, value in zip(rois, values):
        print(f"S16 {roi} layer {SELECTED_LAYERS[roi]}: {value:.16f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_s1()
    plot_s2_s3()
    plot_s4()
    plot_s7()
    plot_s8()
    plot_s9()
    plot_s15_s16()


if __name__ == "__main__":
    main()
