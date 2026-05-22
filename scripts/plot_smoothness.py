import argparse
import numpy as np
import matplotlib.pyplot as plt

from config import PLOTS_DIR
from .analysis_utils import CKPT_GROUPS, METHOD_COLORS, METHOD_LABELS
from .common import MODEL_CKPT
from .get_smoothness import smoothness
from .plot_utils import savefig

DATASET_NAME = "pitcher"
DEFAULT_FWHM = 2.0


def _moving_mean_per_model(results):
    categories = list(results[0].keys())
    moving_cats = [cat for cat in categories if "moving" in cat.lower()]
    if not moving_cats:
        raise ValueError("No moving categories found for smoothness results.")

    model_means = []
    for res in results:
        vals = [res[cat]["model_smoothness"] for cat in moving_cats]
        model_means.append(float(np.mean(vals)))

    human_vals = [results[0][cat]["human_smoothness"] for cat in moving_cats]
    human_moving_mean = float(np.mean(human_vals))

    return model_means, human_moving_mean


def _collect_group_moving(ckpt_names, fwhm_mm):
    results = [
        smoothness(ckpt_name, DATASET_NAME, fwhm_mm=fwhm_mm, resolution_mm=1.0)
        for ckpt_name in ckpt_names
    ]
    return _moving_mean_per_model(results)


def _draw_label(ax, x, y, main, sub=None, main_fs=9, sub_fs=6):
    text_main = ax.text(x, y, main, va="center", ha="left", fontsize=main_fs, color="white")
    if sub:
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = text_main.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        x0, _ = inv.transform((bbox.x0, bbox.y0))
        x1, _ = inv.transform((bbox.x1, bbox.y1))
        x_sub = x + (x1 - x0) + 0.005
        ax.text(x_sub, y, sub, va="center", ha="left", fontsize=sub_fs, color="white")


def _plot_model_sheet_map(t_vals, positions, save_path, title):
    vmax = float(np.nanmax(np.abs(t_vals)))
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=t_vals,
        cmap="bwr",
        s=6,
        marker="s",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
    savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")


def _plot_human_cortex_map(t_vals, save_path, title, view="dorsal"):
    from nilearn import datasets, plotting
    from validate.smoothness import NSD_HIGH

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    t_vals = t_vals.copy()
    t_vals[~NSD_HIGH] = np.nan
    vmax = float(np.nanmax(np.abs(t_vals)))

    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.flat_left,
        stat_map=t_vals[: len(t_vals) // 2],
        hemi="left",
        title=f"{title} (LH)",
        view=view,
        colorbar=True,
        cmap="bwr",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def _plot_smoothness_maps(
    dataset_name,
    ckpt_name,
    category,
    fwhm_mm,
    resolution_mm,
    save_dir,
    mean_categories=False,
):
    from validate import load_transformed_model
    from validate.floc import validate_floc, validate_floc_human
    from models import vit_transform

    save_dir = save_dir / "smoothness_maps"
    save_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device="cuda")
    model.eval()
    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        model_t_vals_dict = validate_floc(
            model,
            vit_transform,
            dataset_names=[dataset_name],
            batch_size=32,
            device="cuda",
        )[0]
        positions = model.smoothed_layer_positions[0].coordinates.cpu().numpy()
        if mean_categories:
            stacked = [vals[0].flatten() for vals in model_t_vals_dict.values()]
            model_t_vals = np.nanmean(np.stack(stacked, axis=0), axis=0)
        else:
            model_t_vals = model_t_vals_dict[category][0].flatten()

    human_t_vals_dict = validate_floc_human([dataset_name])[0]
    if mean_categories:
        stacked = [vals for vals in human_t_vals_dict.values()]
        human_t_vals = np.nanmean(np.stack(stacked, axis=0), axis=0)
        map_label = f"{dataset_name}_mean"
        title_suffix = f"{dataset_name} mean"
    else:
        human_t_vals = human_t_vals_dict[category]
        map_label = category
        title_suffix = category

    _plot_model_sheet_map(
        model_t_vals,
        positions,
        save_dir / f"model_{dataset_name}_{map_label}.png",
        title=f"Model {title_suffix}",
    )
    _plot_human_cortex_map(
        human_t_vals,
        save_dir / f"human_{dataset_name}_{map_label}_lh.png",
        title=f"Human {title_suffix}",
        view="dorsal",
    )


def main():
    groups = [
        ("UNOPTIMIZED", True, 0.0),
        ("UNOPTIMIZED", False, DEFAULT_FWHM),
        ("SWAPOPT", True, 0.0),
        ("SWAPOPT", False, DEFAULT_FWHM),
        ("TDANN", True, 0.0),
        ("TDANN", False, DEFAULT_FWHM),
        ("TopoTransform", True, 0.0),
        ("TopoTransform", False, DEFAULT_FWHM),
        ("ONELAYER", True, 0.0),
        ("ONELAYER", False, DEFAULT_FWHM),
    ]

    means = []
    stds = []
    all_model_means = []
    colors = []
    labels_main = []
    labels_sub = []
    human_moving_mean = None
    group_names = []

    for group_name, no_fmri, fwhm in groups:
        model_means, human_moving_val = _collect_group_moving(CKPT_GROUPS[group_name], fwhm)
        means.append(float(np.mean(model_means)))
        stds.append(float(np.std(model_means)))
        all_model_means.append(model_means)
        if human_moving_mean is None:
            human_moving_mean = human_moving_val
        group_names.append(group_name)
        colors.append(METHOD_COLORS.get(group_name, "#808080"))
        label = METHOD_LABELS.get(group_name, group_name)
        labels_main.append(label)
        labels_sub.append("(no smoothing)" if no_fmri else None)

    y_pos = np.arange(len(groups))

    fig, ax = plt.subplots(1, 1, figsize=(2.6, 2.6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.barh(y_pos, means, height=0.75, color=colors, edgecolor="none")

    ax.set_xlim(0, 1)

    for idx, model_means in enumerate(all_model_means):
        ax.scatter(
            model_means,
            np.full(len(model_means), y_pos[idx]),
            color="black",
            s=8,
            alpha=1,
            zorder=10,
            marker="o",
        )

    label_x = 0.02
    for idx, (main, sub) in enumerate(zip(labels_main, labels_sub)):
        _draw_label(ax, label_x, y_pos[idx], main, sub=sub)

    ax.axvline(human_moving_mean, color="#2E7D32", linestyle="--", linewidth=3.0, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [label if sub is None else f"{label} {sub}" for label, sub in zip(labels_main, labels_sub)],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_xlabel("Spatial autocorrelation (Moran's I)", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    savefig(
        PLOTS_DIR / "smoothness_comparison_bar.svg",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved smoothness comparison plot to {PLOTS_DIR / 'smoothness_comparison_bar.svg'}")

    # grouped plot: no smoothing vs smoothing per method
    combo_means = {}
    combo_stds = {}
    combo_colors = {}
    for (group_name, no_fmri, _), mean, std in zip(groups, means, stds):
        key = group_name
        if key not in combo_means:
            combo_means[key] = {"no_smoothing": None, "smoothing": None}
            combo_stds[key] = {"no_smoothing": None, "smoothing": None}
            combo_colors[key] = METHOD_COLORS.get(group_name, "#808080")
        if no_fmri:
            combo_means[key]["no_smoothing"] = mean
            combo_stds[key]["no_smoothing"] = std
        else:
            combo_means[key]["smoothing"] = mean
            combo_stds[key]["smoothing"] = std

    method_names = list(combo_means.keys())
    method_labels = [METHOD_LABELS.get(name, name) for name in method_names]
    x = np.arange(len(method_names))
    bar_w = 0.36

    fig2, ax2 = plt.subplots(1, 1, figsize=(3.7, 1.9))
    ax2.set_facecolor("white")
    fig2.patch.set_facecolor("white")

    no_vals = [combo_means[m]["no_smoothing"] for m in method_names]
    sm_vals = [combo_means[m]["smoothing"] for m in method_names]
    no_err = [combo_stds[m]["no_smoothing"] for m in method_names]
    sm_err = [combo_stds[m]["smoothing"] for m in method_names]
    base_colors = [combo_colors[m] for m in method_names]

    no_mask = np.array([v is not None for v in no_vals])
    sm_mask = np.array([v is not None for v in sm_vals])

    ax2.bar(
        x[no_mask] - bar_w / 2,
        np.array(no_vals, dtype=float)[no_mask],
        width=bar_w,
        color=np.array(base_colors, dtype=object)[no_mask],
        alpha=0.45,
        edgecolor="none",
        label="No smoothing",
    )
    ax2.bar(
        x[sm_mask] + bar_w / 2,
        np.array(sm_vals, dtype=float)[sm_mask],
        width=bar_w,
        color=np.array(base_colors, dtype=object)[sm_mask],
        alpha=1.0,
        edgecolor="none",
        label="Smoothing",
    )

    ax2.errorbar(
        x[no_mask] - bar_w / 2,
        np.array(no_vals, dtype=float)[no_mask],
        yerr=np.array(no_err, dtype=float)[no_mask],
        fmt="none",
        ecolor="#333333",
        elinewidth=1,
        capsize=2,
    )
    ax2.errorbar(
        x[sm_mask] + bar_w / 2,
        np.array(sm_vals, dtype=float)[sm_mask],
        yerr=np.array(sm_err, dtype=float)[sm_mask],
        fmt="none",
        ecolor="#333333",
        elinewidth=1,
        capsize=2,
    )

    ax2.axhline(human_moving_mean, color="#2E7D32", linestyle="--", linewidth=2.5, zorder=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, fontsize=8, rotation=20, ha="right")
    ax2.set_ylabel("Spatial autocorrelation (Moran's I)", fontsize=9)
    ax2.set_xlim(-0.6, len(method_names) - 0.4)
    ax2.tick_params(axis="both", which="major", labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")

    plt.tight_layout()
    savefig(
        PLOTS_DIR / "smoothness_comparison_bar_grouped.svg",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved grouped smoothness comparison plot to {PLOTS_DIR / 'smoothness_comparison_bar_grouped.svg'}")

    print("\nMODEL vs HUMAN SMOOTHNESS COMPARISON (moving)")
    for name, mean, std in zip(group_names, means, stds):
        diff = mean - human_moving_mean
        print(f"{name:12s} - Diff mean +/- std: {diff:+.4f} +/- {std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-maps", action="store_true", help="Plot spatial maps for one category.")
    parser.add_argument("--map-category", type=str, default=None, help="Category name to plot (e.g., Faces_moving).")
    parser.add_argument("--map-dataset", type=str, default=DATASET_NAME, help="Dataset name for maps.")
    parser.add_argument("--map-ckpt", type=str, default=MODEL_CKPT, help="Checkpoint name for model map.")
    parser.add_argument("--map-fwhm", type=float, default=DEFAULT_FWHM, help="FWHM for model smoothing.")
    parser.add_argument("--map-resolution", type=float, default=1.0, help="Resolution for model smoothing.")
    args = parser.parse_args()

    main()

    if args.plot_maps:
        mean_categories = False
        if args.map_dataset == "pitcher" and args.map_category is None:
            mean_categories = True
        if args.map_category is None and not mean_categories:
            sample = smoothness(
                args.map_ckpt,
                args.map_dataset,
                fwhm_mm=args.map_fwhm,
                resolution_mm=args.map_resolution,
            )
            categories = [cat for cat in sample.keys() if "moving" in cat.lower()]
            if not categories:
                categories = list(sample.keys())
            args.map_category = categories[0]

        _plot_smoothness_maps(
            dataset_name=args.map_dataset,
            ckpt_name=args.map_ckpt,
            category=args.map_category,
            fwhm_mm=args.map_fwhm,
            resolution_mm=args.map_resolution,
            save_dir=PLOTS_DIR,
            mean_categories=mean_categories,
        )
