import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import PLOTS_DIR
from models import clip_transform
from models import vit_transform
from utils import cached
from validate import load_transformed_model
from validate.floc.utils import CategoryDataset
from validate.floc.utils import run_features

from .common import MODEL_CKPT
from .get_localizers import localizers
from .plot_utils import ensure_dir
from .plot_utils import to_numpy


OBJECT100_DIR = Path("/mnt/scratch/ytang/datasets/konkle/OBJECT100Database")
IMAGE_DIRNAME = "HighRes 100"
SIZE_TABLE = "sizerank.csv"

FILENAME_TO_SIZE_NAME = {
    "coffeemaker": "coffemaker",
    "germanshepard": "germanshephard",
    "goldfish": "fish",
    "hair clip": "hairclip",
    "refrigerator": "refridgerator",
}

DEFAULT_EXCLUDED_LIVING = {
    "clown",
    "cow",
    "germanshepard",
    "goldfish",
    "kitten",
    "toddler",
    "trafficcop",
    "tree",
}


def _get_input_transform(model):
    return clip_transform if model.__class__.__name__ == "TopoTransformedCLIP" else vit_transform


def _get_layer_positions(model):
    if model.smoothing:
        return [pos.coordinates.detach().cpu().numpy() for pos in model.smoothed_layer_positions]
    return [pos.coordinates.detach().cpu().numpy() for pos in model.layer_positions]


def _safe_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))


def _load_object100_metadata(object100_dir, include_living=False):
    object100_dir = Path(object100_dir)
    image_dir = object100_dir / IMAGE_DIRNAME
    size_table = pd.read_csv(object100_dir / SIZE_TABLE)
    size_table = size_table.set_index("Object", drop=False)

    rows = []
    missing = []
    excluded = []
    for path in sorted(image_dir.glob("*.jpg")):
        image_name = path.stem
        if not include_living and image_name in DEFAULT_EXCLUDED_LIVING:
            excluded.append(image_name)
            continue

        size_name = FILENAME_TO_SIZE_NAME.get(image_name, image_name)
        if size_name not in size_table.index:
            missing.append(image_name)
            continue

        row = size_table.loc[size_name]
        rows.append(
            {
                "image_path": str(path),
                "image_name": image_name,
                "size_name": size_name,
                "size_rank": float(row["SizeRank"]),
                "diag_cm": float(row["DiagSize_cm"]),
                "log_cm": float(row["Log_cm"]),
            }
        )

    if missing:
        raise ValueError(f"Images missing from size table: {missing}")
    if not rows:
        raise ValueError(f"No usable OBJECT100 images found in {image_dir}")

    meta = pd.DataFrame(rows)
    meta.attrs["excluded_living"] = excluded
    return meta


def _feature_cache_key(
    checkpoint_name,
    object100_dir,
    include_living,
    fwhm_mm,
    resolution_mm,
    frames_per_video,
    video_fps,
):
    pieces = [
        checkpoint_name,
        str(Path(object100_dir).resolve()),
        str(include_living),
        str(fwhm_mm),
        str(resolution_mm),
        str(frames_per_video),
        str(video_fps),
    ]
    return hashlib.md5("|".join(pieces).encode()).hexdigest()[:12]


def _extract_features_uncached(
    checkpoint_name,
    object100_dir,
    include_living,
    batch_size,
    device,
    fwhm_mm,
    resolution_mm,
    frames_per_video,
    video_fps,
):
    meta = _load_object100_metadata(object100_dir, include_living=include_living)
    model, _epoch = load_transformed_model(checkpoint_name=checkpoint_name, device=device)
    transform = _get_input_transform(model)
    file_infos = [(path, "object100") for path in meta["image_path"]]
    dataset = CategoryDataset(
        file_infos,
        transform=transform,
        frames_per_video=frames_per_video,
        video_fps=video_fps,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=max(0, min(8, int(batch_size / 2))),
        shuffle=False,
        pin_memory=device.startswith("cuda"),
    )

    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        features, _targets = run_features(model, loader, device)
        features = [np.mean(feat, axis=1) for feat in features]
        positions = _get_layer_positions(model)

    return {
        "features": features,
        "positions": positions,
        "metadata": meta,
        "excluded_living": meta.attrs.get("excluded_living", []),
    }


def extract_features(
    checkpoint_name,
    object100_dir,
    include_living,
    batch_size,
    device,
    fwhm_mm,
    resolution_mm,
    frames_per_video,
    video_fps,
    rerun=False,
):
    key = _feature_cache_key(
        checkpoint_name,
        object100_dir,
        include_living,
        fwhm_mm,
        resolution_mm,
        frames_per_video,
        video_fps,
    )
    cache_name = f"konkle_object100_size_features_{key}"
    return cached(cache_name, persistent=True, rerun=rerun)(
        _extract_features_uncached
    )(
        checkpoint_name,
        object100_dir,
        include_living,
        batch_size,
        device,
        fwhm_mm,
        resolution_mm,
        frames_per_video,
        video_fps,
    )


def _flatten_layers(features, positions):
    flat_features = [feat.reshape(feat.shape[0], -1) for feat in features]
    flat_positions = [np.asarray(pos).reshape(-1, 2) for pos in positions]
    return np.concatenate(flat_features, axis=1), np.concatenate(flat_positions, axis=0)


def _flatten_tvals(t_vals):
    if not isinstance(t_vals, (list, tuple)):
        t_vals = [t_vals]
    return np.concatenate([np.asarray(t_val).reshape(-1) for t_val in t_vals])


def _object_roi_mask(checkpoint_name, object_top_percent, fwhm_mm, resolution_mm, device):
    t_vals_dict, _p_vals_dict, layer_positions = localizers(
        checkpoint_name,
        dataset_names=["konkle"],
        device=device,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
        ret_merged=True,
    )
    if "animal_vs_object" not in t_vals_dict:
        raise ValueError("Konkle localizer did not return animal_vs_object.")

    object_score = -_flatten_tvals(t_vals_dict["animal_vs_object"])
    finite = np.isfinite(object_score)
    cutoff = np.nanpercentile(object_score[finite], 100.0 - object_top_percent)
    mask = finite & (object_score >= cutoff)
    positions = np.concatenate(
        [to_numpy(pos).reshape(-1, 2) for pos in layer_positions],
        axis=0,
    )
    return mask, object_score, positions, cutoff


def _unit_correlations(features, target):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    y = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x * x, axis=0)) * np.sqrt(np.nansum(y * y))
    out = np.full(x.shape[1], np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = np.nansum(x[:, valid] * y[:, None], axis=0) / denom[valid]
    return out


def _decode_target(features, target, seed, n_splits=5):
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if features.shape[1] < 1 or features.shape[0] < 4:
        return {"pearson_r": np.nan, "r2": np.nan}

    cv = KFold(
        n_splits=min(n_splits, features.shape[0]),
        shuffle=True,
        random_state=seed,
    )
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-6, 6, 13)),
    )
    pred = cross_val_predict(model, features, target, cv=cv)
    if np.std(pred) == 0 or np.std(target) == 0:
        pearson_r = np.nan
    else:
        pearson_r = stats.pearsonr(target, pred).statistic
    return {
        "pearson_r": float(pearson_r),
        "r2": float(r2_score(target, pred)),
    }


def _stripe_masks(positions, roi_mask, n_stripes):
    x = positions[:, 0]
    x_roi = x[roi_mask]
    edges = np.linspace(float(np.min(x_roi)), float(np.max(x_roi)), n_stripes + 1)
    masks = []
    for idx in range(n_stripes):
        if idx == n_stripes - 1:
            mask = roi_mask & (x >= edges[idx]) & (x <= edges[idx + 1])
        else:
            mask = roi_mask & (x >= edges[idx]) & (x < edges[idx + 1])
        masks.append((idx, 0.5 * (edges[idx] + edges[idx + 1]), mask))
    return masks


def analyze_object_size(
    features,
    positions,
    roi_mask,
    metadata,
    n_stripes,
    seed,
    min_units_per_stripe,
):
    targets = {
        "linear_cm": metadata["diag_cm"].to_numpy(dtype=np.float64),
        "log_cm": metadata["log_cm"].to_numpy(dtype=np.float64),
    }
    full_features = features[:, roi_mask]
    summary_rows = []

    for target_name, target in targets.items():
        scores = _decode_target(full_features, target, seed=seed)
        summary_rows.append(
            {
                "analysis": "full_object_roi",
                "target": target_name,
                "stripe": -1,
                "x_mid": np.nan,
                "n_units": int(np.sum(roi_mask)),
                **scores,
            }
        )

    for stripe, x_mid, stripe_mask in _stripe_masks(positions, roi_mask, n_stripes):
        n_units = int(np.sum(stripe_mask))
        if n_units < min_units_per_stripe:
            for target_name in targets:
                summary_rows.append(
                    {
                        "analysis": "x_stripe",
                        "target": target_name,
                        "stripe": stripe,
                        "x_mid": x_mid,
                        "n_units": n_units,
                        "pearson_r": np.nan,
                        "r2": np.nan,
                    }
                )
            continue

        stripe_features = features[:, stripe_mask]
        for target_name, target in targets.items():
            scores = _decode_target(stripe_features, target, seed=seed)
            summary_rows.append(
                {
                    "analysis": "x_stripe",
                    "target": target_name,
                    "stripe": stripe,
                    "x_mid": x_mid,
                    "n_units": n_units,
                    **scores,
                }
            )

    unit_r_log = _unit_correlations(features, targets["log_cm"])
    stripe_gradient_rows = []
    for stripe, x_mid, stripe_mask in _stripe_masks(positions, roi_mask, n_stripes):
        vals = unit_r_log[stripe_mask]
        vals = vals[np.isfinite(vals)]
        stripe_gradient_rows.append(
            {
                "stripe": stripe,
                "x_mid": x_mid,
                "n_units": int(vals.size),
                "mean_log_size_r": float(np.mean(vals)) if vals.size else np.nan,
                "sem_log_size_r": (
                    float(stats.sem(vals)) if vals.size > 1 else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(stripe_gradient_rows),
        unit_r_log,
    )


def _rect_geometry(positions):
    xs = np.unique(positions[:, 0])
    ys = np.unique(positions[:, 1])
    dx = float(np.median(np.diff(np.sort(xs)))) if len(xs) > 1 else 1.0
    dy = float(np.median(np.diff(np.sort(ys)))) if len(ys) > 1 else 1.0
    rects = [Rectangle((x - dx / 2, y - dy / 2), dx, dy) for x, y in positions]
    return rects, dx, dy


def plot_results(
    positions,
    roi_mask,
    object_score,
    unit_r_log,
    decode_df,
    gradient_df,
    metadata,
    output_path,
    object_top_percent,
):
    stripe_centers = gradient_df["x_mid"].to_numpy(dtype=float)
    stripe_labels = [f"{x:.1f}" for x in stripe_centers]
    decode_colors = {"linear_cm": "#4C78A8", "log_cm": "#E4572E"}

    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)

    def panel_label(ax, label):
        ax.text(
            -0.12,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig = plt.figure(figsize=(14.2, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[0.6, 1.55],
        height_ratios=[0.62, 1.0, 1.0],
        wspace=0.02,
        hspace=0.02,
    )
    ax_decode_full = fig.add_subplot(grid[0, 0])
    ax_gradient = fig.add_subplot(grid[1, 0])
    ax_decode_stripes = fig.add_subplot(grid[2, 0])
    ax_map = fig.add_subplot(grid[:, 1])

    map_positions = positions[:, [1, 0]].copy()
    map_positions[:, 0] = np.max(map_positions[:, 0]) - map_positions[:, 0]
    rects, dx, dy = _rect_geometry(map_positions)
    ax_map.add_collection(
        PatchCollection(rects, facecolor="#EEEEEE", edgecolor="none", alpha=0.75, rasterized=True)
    )
    vmax = float(np.nanpercentile(np.abs(unit_r_log[roi_mask]), 98))
    vmax = max(vmax, 0.05)
    roi_rects = [rect for rect, keep in zip(rects, roi_mask) if keep]
    roi_vals = unit_r_log[roi_mask]
    collection = PatchCollection(
        roi_rects,
        array=roi_vals,
        cmap="RdBu_r",
        norm=Normalize(vmin=-vmax, vmax=vmax),
        edgecolor="none",
        rasterized=True,
    )
    ax_map.add_collection(collection)
    ax_map.set_aspect("equal", "box")
    ax_map.set_xlim(float(map_positions[:, 0].min()) - dx, float(map_positions[:, 0].max()) + dx)
    ax_map.set_ylim(float(map_positions[:, 1].min()) - dy, float(map_positions[:, 1].max()) + dy)
    ax_map.set_title(
        f"Log-size responsiveness in top {object_top_percent:g}% object ROI",
        loc="left",
        fontweight="bold",
    )
    ax_map.set_xlabel("")
    ax_map.set_ylabel("sheet x")
    ax_map.tick_params(length=0)
    panel_label(ax_map, "D")
    divider = make_axes_locatable(ax_map)
    cbar_ax = divider.append_axes("bottom", size="4%", pad=0.08)
    cbar = fig.colorbar(collection, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("unit r(response, log cm)")
    cbar.outline.set_visible(False)

    ax_gradient.errorbar(
        gradient_df["x_mid"],
        gradient_df["mean_log_size_r"],
        yerr=gradient_df["sem_log_size_r"],
        marker="o",
        markersize=5,
        color="#2F2F2F",
        ecolor="#7A7A7A",
        elinewidth=1.2,
        capsize=3,
        linewidth=2.3,
    )
    ax_gradient.axhline(0, color="#9E9E9E", linewidth=1)
    ax_gradient.set_xticks(stripe_centers)
    ax_gradient.set_xticklabels(stripe_labels)
    ax_gradient.set_title("Size responsiveness across sheet x", loc="left", fontweight="bold")
    ax_gradient.set_xlabel("stripe center x")
    ax_gradient.set_ylabel("mean unit r")
    style_axis(ax_gradient)
    panel_label(ax_gradient, "B")

    full = decode_df[decode_df["analysis"] == "full_object_roi"].copy()
    labels = ["linear_cm", "log_cm"]
    full_scores = [
        float(full.loc[full["target"] == label, "pearson_r"].iloc[0])
        for label in labels
    ]
    bars = ax_decode_full.bar(
        ["linear", "log"],
        full_scores,
        color=[decode_colors[label] for label in labels],
        width=0.62,
    )
    for bar, score in zip(bars, full_scores):
        ax_decode_full.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.035 if score >= 0 else score - 0.06,
            f"{score:.2f}",
            ha="center",
            va="bottom" if score >= 0 else "top",
            fontsize=9,
        )
    ax_decode_full.axhline(0, color="#9E9E9E", linewidth=1)
    ax_decode_full.set_ylim(0.0, 1.0)
    ax_decode_full.set_ylabel("CV Pearson r")
    ax_decode_full.set_title(
        f"Full object ROI decoding, n={len(metadata)} images",
        loc="left",
        fontweight="bold",
    )
    style_axis(ax_decode_full)
    panel_label(ax_decode_full, "A")

    stripes = decode_df[decode_df["analysis"] == "x_stripe"].copy()
    for label in labels:
        sub = stripes[stripes["target"] == label].sort_values("stripe")
        ax_decode_stripes.plot(
            sub["x_mid"],
            sub["pearson_r"],
            marker="o",
            markersize=5,
            linewidth=2.3,
            color=decode_colors[label],
            label="linear size" if label == "linear_cm" else "log size",
        )
    ax_decode_stripes.axhline(0, color="#9E9E9E", linewidth=1)
    ax_decode_stripes.set_ylim(-1.0, 1.0)
    ax_decode_stripes.set_xticks(stripe_centers)
    ax_decode_stripes.set_xticklabels(stripe_labels)
    ax_decode_stripes.set_title("Decodability across x stripes", loc="left", fontweight="bold")
    ax_decode_stripes.set_xlabel("stripe center x, anterior to posterior")
    ax_decode_stripes.set_ylabel("CV Pearson r")
    ax_decode_stripes.legend(frameon=False, ncol=2, loc="lower left")
    style_axis(ax_decode_stripes)
    panel_label(ax_decode_stripes, "C")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="OBJECT100 size-gradient and decoding analysis within the Konkle object ROI."
    )
    parser.add_argument("--ckpt", default=MODEL_CKPT, help="Checkpoint name.")
    parser.add_argument(
        "--object100-dir",
        default=str(OBJECT100_DIR),
        help="OBJECT100Database directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PLOTS_DIR / "konkle_object_size"),
        help="Output directory.",
    )
    parser.add_argument("--object-top-percent", type=float, default=10.0)
    parser.add_argument("--n-stripes", type=int, default=6)
    parser.add_argument("--min-units-per-stripe", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fwhm-mm", type=float, default=2.0)
    parser.add_argument("--resolution-mm", type=float, default=1.0)
    parser.add_argument("--frames-per-video", type=int, default=24)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--include-living", action="store_true")
    parser.add_argument("--rerun-features", action="store_true")
    args = parser.parse_args()

    if not (0 < args.object_top_percent <= 100):
        raise ValueError("--object-top-percent must be in (0, 100].")

    out_dir = ensure_dir(args.out_dir)
    feature_result = extract_features(
        checkpoint_name=args.ckpt,
        object100_dir=args.object100_dir,
        include_living=args.include_living,
        batch_size=args.batch_size,
        device=args.device,
        fwhm_mm=args.fwhm_mm,
        resolution_mm=args.resolution_mm,
        frames_per_video=args.frames_per_video,
        video_fps=args.video_fps,
        rerun=args.rerun_features,
    )
    features, positions = _flatten_layers(
        feature_result["features"],
        feature_result["positions"],
    )
    metadata = feature_result["metadata"]

    roi_mask, object_score, localizer_positions, cutoff = _object_roi_mask(
        args.ckpt,
        args.object_top_percent,
        args.fwhm_mm,
        args.resolution_mm,
        args.device,
    )
    if features.shape[1] != roi_mask.shape[0]:
        raise ValueError(
            "Feature and localizer shapes do not match: "
            f"{features.shape[1]} units vs {roi_mask.shape[0]} localizer values."
        )
    if positions.shape != localizer_positions.shape or not np.allclose(
        positions, localizer_positions
    ):
        print("[WARN] Feature positions differ from localizer positions; using feature positions.")

    decode_df, gradient_df, unit_r_log = analyze_object_size(
        features,
        positions,
        roi_mask,
        metadata,
        n_stripes=args.n_stripes,
        seed=args.seed,
        min_units_per_stripe=args.min_units_per_stripe,
    )

    stem = _safe_name(Path(args.ckpt).stem)
    plot_path = out_dir / f"konkle_object_size_{stem}.svg"
    decode_path = out_dir / f"konkle_object_size_decode_{stem}.csv"
    gradient_path = out_dir / f"konkle_object_size_gradient_{stem}.csv"
    meta_path = out_dir / f"konkle_object_size_metadata_{stem}.csv"

    plot_results(
        positions,
        roi_mask,
        object_score,
        unit_r_log,
        decode_df,
        gradient_df,
        metadata,
        plot_path,
        args.object_top_percent,
    )
    decode_df.to_csv(decode_path, index=False)
    gradient_df.to_csv(gradient_path, index=False)
    metadata.to_csv(meta_path, index=False)

    print(f"Included OBJECT100 images: {len(metadata)}")
    if feature_result["excluded_living"]:
        print(f"Excluded living/animate images: {feature_result['excluded_living']}")
    print(f"Object ROI cutoff: {cutoff:.3f}; units: {int(np.sum(roi_mask))}")
    print("\nFull object ROI decoding:")
    print(
        decode_df[decode_df["analysis"] == "full_object_roi"][
            ["target", "n_units", "pearson_r", "r2"]
        ].to_string(index=False)
    )
    print("\nStripe decoding:")
    print(
        decode_df[decode_df["analysis"] == "x_stripe"][
            ["target", "stripe", "x_mid", "n_units", "pearson_r", "r2"]
        ].to_string(index=False)
    )
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved decode CSV: {decode_path}")
    print(f"Saved gradient CSV: {gradient_path}")
    print(f"Saved metadata CSV: {meta_path}")


if __name__ == "__main__":
    main()
