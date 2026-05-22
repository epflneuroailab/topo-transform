import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from config import PLOTS_DIR

from .common import MODEL_CKPT
from .get_localizers import localizers
from .plot_utils import ensure_dir, to_numpy


def _filter_konkle(t_vals_dict):
    keep = ["animal_vs_object", "big_vs_small"]
    filtered = {k: v for k, v in t_vals_dict.items() if k in keep}
    if "big_vs_small" in filtered:
        filtered["small_vs_big"] = [-vals for vals in filtered["big_vs_small"]]
        del filtered["big_vs_small"]
    return filtered


def _plot_konkle_tvals(t_vals_dict, layer_positions, store_dir, prefix="", suffix="", dpi=300):
    green_gray_pink = LinearSegmentedColormap.from_list(
        "green_gray_pink", ["#25CB10", "#BDBDBD", "#E22F7E"]
    )
    blue_gray_orange = LinearSegmentedColormap.from_list(
        "blue_gray_orange", ["#003BF9", "#BDBDBD", "#FB9332"]
    )

    cmap_map = {
        "animal_vs_object": green_gray_pink,
        "small_vs_big": blue_gray_orange,
    }

    for cat_name, t_vals_list in t_vals_dict.items():
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]

        n_cols = len(t_vals_list)
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))
        if n_cols == 1:
            axes = [axes]

        all_t_vals = np.concatenate([t_vals.flatten() for t_vals in t_vals_list])
        vmax_global = float(np.abs(all_t_vals).max())
        norm_global = Normalize(vmin=-vmax_global, vmax=vmax_global)
        print(vmax_global)

        cmap = cmap_map.get(cat_name, "bwr")

        for layer_idx, t_vals in enumerate(t_vals_list):
            pos = to_numpy(layer_positions[layer_idx])
            t_vals_flat = t_vals.flatten()

            xs = np.unique(pos[:, 0])
            ys = np.unique(pos[:, 1])
            dx = float(np.median(np.diff(np.sort(xs)))) if len(xs) > 1 else 1.0
            dy = float(np.median(np.diff(np.sort(ys)))) if len(ys) > 1 else 1.0

            rects = [
                Rectangle((x - dx / 2, y - dy / 2), dx, dy)
                for x, y in pos
            ]
            collection = PatchCollection(
                rects,
                array=t_vals_flat,
                cmap=cmap,
                norm=norm_global,
                edgecolor="none",
            )
            axes[layer_idx].add_collection(collection)
            axes[layer_idx].set_aspect("equal", "box")

            x_min, x_max = float(pos[:, 0].min()), float(pos[:, 0].max())
            y_min, y_max = float(pos[:, 1].min()), float(pos[:, 1].max())
            axes[layer_idx].set_xlim(x_min - dx / 2, x_max + dx / 2)
            axes[layer_idx].set_ylim(y_min - dy / 2, y_max + dy / 2)
            axes[layer_idx].set_axis_off()

        cbar_ax = fig.add_axes([0.375, -0.07, 0.25, 0.03])
        fig.colorbar(collection, cax=cbar_ax, orientation="horizontal", label="t-statistic")
        plt.tight_layout()
        plt.savefig(f"{store_dir}/{prefix}tvals_{cat_name}{suffix}.png", dpi=dpi, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Konkle localizers (animal vs object, big vs small) on the model sheet."
    )
    parser.add_argument("--ckpt", default=MODEL_CKPT, help="Checkpoint name.")
    parser.add_argument(
        "--out-dir",
        default=str(PLOTS_DIR / "konkle_localizers"),
        help="Output directory for plots.",
    )
    parser.add_argument("--topk-percent", type=float, default=100.0, help="Top-k percent to plot.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures.")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    t_vals_dict, _p_vals_dict, layer_positions = localizers(
        args.ckpt, dataset_names=["konkle"], ret_merged=True
    )
    t_vals_dict = _filter_konkle(t_vals_dict)
    if not t_vals_dict:
        raise ValueError("No Konkle localizers found. Did the localizer run?")

    if args.topk_percent < 100:
        raise ValueError("topk_percent < 100 not supported in this script yet.")

    _plot_konkle_tvals(
        t_vals_dict,
        layer_positions,
        store_dir=str(out_dir),
        prefix="konkle_",
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
