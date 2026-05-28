import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from config import PLOTS_DIR
from .common import LOCALIZER_P_THRESHOLD
from .common import LOCALIZER_T_THRESHOLD
from .common import MODEL_CKPT
from .get_localizers import localizers
from .plot_localizers import _infer_tile_size
from .plot_utils import ensure_dir
from .plot_utils import savefig
from .plot_utils import to_numpy


CLASS_ORDER = [
    "adult",
    "child",
    "body",
    "limb",
    "car",
    "instrument",
    "corridor",
    "house",
    "word",
    "number",
]

CLASS_COLORS = {
    "adult": "#D73027",
    "child": "#FC8D59",
    "body": "#1A9850",
    "limb": "#91CF60",
    "car": "#4575B4",
    "instrument": "#74ADD1",
    "corridor": "#FEE08B",
    "house": "#D9A441",
    "word": "#7B3294",
    "number": "#C2A5CF",
}


def _rects_for_positions(positions, tile_size):
    dx, dy = tile_size
    return [
        Rectangle((x - dx / 2, y - dy / 2), dx, dy)
        for x, y in positions
    ]


def _style_sheet_axis(ax, positions, tile_size):
    dx, dy = tile_size
    ax.set_aspect("equal", "box")
    ax.set_xlim(float(np.min(positions[:, 0])) - dx / 2, float(np.max(positions[:, 0])) + dx / 2)
    ax.set_ylim(float(np.min(positions[:, 1])) - dy / 2, float(np.max(positions[:, 1])) + dy / 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _significant_mask(t_vals, p_vals, p_threshold, t_threshold):
    return (np.asarray(p_vals).reshape(-1) < p_threshold) & (
        np.asarray(t_vals).reshape(-1) > t_threshold
    )


def plot_vpnl_detail_classes(
    t_vals_dict,
    p_vals_dict,
    layer_positions,
    out_dir,
    p_threshold=LOCALIZER_P_THRESHOLD,
    t_threshold=LOCALIZER_T_THRESHOLD,
    dpi=300,
):
    out_dir = ensure_dir(out_dir)
    class_names = [name for name in CLASS_ORDER if name in t_vals_dict]
    class_names += sorted(set(t_vals_dict) - set(class_names))
    n_layers = len(layer_positions)
    positions_by_layer = [to_numpy(pos) for pos in layer_positions]
    tile_sizes = [_infer_tile_size(pos) for pos in positions_by_layer]

    fig, axes = plt.subplots(
        1,
        n_layers,
        figsize=(3.2 * n_layers, 3.4),
        constrained_layout=True,
    )
    if n_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        positions = positions_by_layer[layer_idx]
        tile_size = tile_sizes[layer_idx]
        background = PatchCollection(
            _rects_for_positions(positions, tile_size),
            facecolor="#EFEFEF",
            edgecolor="none",
            alpha=0.8,
            rasterized=True,
        )
        ax.add_collection(background)
        for class_name in class_names:
            mask = _significant_mask(
                t_vals_dict[class_name][layer_idx],
                p_vals_dict[class_name][layer_idx],
                p_threshold,
                t_threshold,
            )
            if not np.any(mask):
                continue
            rects = _rects_for_positions(positions[mask], tile_size)
            ax.add_collection(
                PatchCollection(
                    rects,
                    facecolor=CLASS_COLORS.get(class_name, "#666666"),
                    edgecolor="none",
                    alpha=0.95,
                    rasterized=True,
                )
            )
        _style_sheet_axis(ax, positions, tile_size)
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=9)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=CLASS_COLORS.get(class_name, "#666666"),
            markeredgecolor="none",
            markersize=7,
        )
        for class_name in class_names
    ]
    axes[0].legend(
        handles,
        class_names,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(-0.48, 0.5),
        borderaxespad=0,
    )
    overlay_path = out_dir / "vpnl_detail_classes_overlay.svg"
    savefig(overlay_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay: {overlay_path}")

    for class_name in class_names:
        fig, axes = plt.subplots(
            1,
            n_layers,
            figsize=(3.0 * n_layers, 3.0),
            constrained_layout=True,
        )
        if n_layers == 1:
            axes = [axes]
        for layer_idx, ax in enumerate(axes):
            positions = positions_by_layer[layer_idx]
            tile_size = tile_sizes[layer_idx]
            mask = _significant_mask(
                t_vals_dict[class_name][layer_idx],
                p_vals_dict[class_name][layer_idx],
                p_threshold,
                t_threshold,
            )
            ax.add_collection(
                PatchCollection(
                    _rects_for_positions(positions, tile_size),
                    facecolor="#EFEFEF",
                    edgecolor="none",
                    alpha=0.8,
                    rasterized=True,
                )
            )
            if np.any(mask):
                ax.add_collection(
                    PatchCollection(
                        _rects_for_positions(positions[mask], tile_size),
                        facecolor=CLASS_COLORS.get(class_name, "#666666"),
                        edgecolor="none",
                        alpha=0.95,
                        rasterized=True,
                    )
                )
            _style_sheet_axis(ax, positions, tile_size)
            ax.set_title(f"Layer {layer_idx + 1}", fontsize=9)
        fig.suptitle(class_name, fontsize=11, fontweight="bold")
        class_path = out_dir / f"vpnl_detail_class_{class_name}.svg"
        savefig(class_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved class map: {class_path}")


def plot_vpnl_detail_tvals(
    t_vals_dict,
    layer_positions,
    out_dir,
    dpi=300,
):
    out_dir = ensure_dir(out_dir)
    class_names = [name for name in CLASS_ORDER if name in t_vals_dict]
    class_names += sorted(set(t_vals_dict) - set(class_names))
    positions_by_layer = [to_numpy(pos) for pos in layer_positions]
    tile_sizes = [_infer_tile_size(pos) for pos in positions_by_layer]
    vmax = max(
        float(np.nanmax(np.abs(to_numpy(t_vals_dict[class_name][layer_idx]))))
        for class_name in class_names
        for layer_idx in range(len(layer_positions))
    )
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = "RdBu_r"

    for class_name in class_names:
        fig, axes = plt.subplots(
            1,
            len(layer_positions),
            figsize=(3.1 * len(layer_positions), 3.1),
            constrained_layout=True,
        )
        if len(layer_positions) == 1:
            axes = [axes]
        collection = None
        for layer_idx, ax in enumerate(axes):
            positions = positions_by_layer[layer_idx]
            tile_size = tile_sizes[layer_idx]
            vals = to_numpy(t_vals_dict[class_name][layer_idx]).reshape(-1)
            collection = PatchCollection(
                _rects_for_positions(positions, tile_size),
                array=vals,
                cmap=cmap,
                norm=norm,
                edgecolor="none",
                rasterized=True,
            )
            ax.add_collection(collection)
            _style_sheet_axis(ax, positions, tile_size)
            ax.set_title(f"Layer {layer_idx + 1}", fontsize=9)
        fig.suptitle(f"{class_name} t-values", fontsize=11, fontweight="bold")
        fig.colorbar(collection, ax=axes, shrink=0.78, pad=0.015, label="t-value")
        tval_path = out_dir / f"vpnl_detail_tvals_{class_name}.svg"
        savefig(tval_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved t-value map: {tval_path}")

    print(f"Shared t-value color scale: [{-vmax:.3f}, {vmax:.3f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Plot VPNL raw filename-prefix classes as one-vs-rest localizers."
    )
    parser.add_argument("--ckpt", default=MODEL_CKPT)
    parser.add_argument("--out-dir", default=str(PLOTS_DIR / "vpnl_detail_localizers"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fwhm-mm", type=float, default=2.0)
    parser.add_argument("--resolution-mm", type=float, default=1.0)
    parser.add_argument("--p-threshold", type=float, default=LOCALIZER_P_THRESHOLD)
    parser.add_argument("--t-threshold", type=float, default=LOCALIZER_T_THRESHOLD)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    stem = Path(args.ckpt).stem.replace(".", "_")
    out_dir = ensure_dir(Path(args.out_dir) / stem)
    t_vals_dict, p_vals_dict, layer_positions = localizers(
        args.ckpt,
        dataset_names=["vpnl_detail_classes"],
        device=args.device,
        fwhm_mm=args.fwhm_mm,
        resolution_mm=args.resolution_mm,
        ret_merged=True,
    )
    plot_vpnl_detail_classes(
        t_vals_dict,
        p_vals_dict,
        layer_positions,
        out_dir,
        p_threshold=args.p_threshold,
        t_threshold=args.t_threshold,
        dpi=args.dpi,
    )
    plot_vpnl_detail_tvals(
        t_vals_dict,
        layer_positions,
        out_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
