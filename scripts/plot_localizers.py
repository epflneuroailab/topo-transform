from config import PLOTS_DIR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .common import LOCALIZER_P_THRESHOLD
from .common import LOCALIZER_T_THRESHOLD
from .common import MODEL_CKPT
from .common import all_roi_colors
from .common import roi_groups
from .get_localizers import localizers
from .localizer_registry import get_roi_t_threshold
from .plot_utils import ensure_dir, savefig, to_numpy
from .analysis_utils import CKPT_GROUPS


def _infer_tile_size(pos):
    xs = np.unique(pos[:, 0])
    ys = np.unique(pos[:, 1])
    dx = np.median(np.diff(np.sort(xs))) if len(xs) > 1 else 1.0
    dy = np.median(np.diff(np.sort(ys))) if len(ys) > 1 else 1.0
    return float(dx), float(dy)


def scatter_roi(ax, pos_filtered, color, label, tile_size):
    dx, dy = tile_size
    rects = [
        Rectangle((x - dx / 2, y - dy / 2), dx, dy)
        for x, y in pos_filtered
    ]
    if rects:
        ax.add_collection(
            PatchCollection(rects, facecolor=color, edgecolor='none', alpha=1)
        )
    if label:
        ax.scatter([], [], color=color, s=30, label=label, marker='s', edgecolors='none')

def plot_all_rois(
    t_vals_dict, 
    p_vals_dict, 
    layer_positions, 
    store_dir, 
    figsize_per_panel=3, 
    prefix='', 
    suffix='', 
    p_threshold=LOCALIZER_P_THRESHOLD, 
    t_threshold=LOCALIZER_T_THRESHOLD, 
    dpi=400
):
    
    ensure_dir(store_dir)

    # Get number of layers from first ROI
    first_roi = next(iter(p_vals_dict.keys()))
    n_layers = len(p_vals_dict[first_roi])
    
    tile_sizes = [_infer_tile_size(to_numpy(lp)) for lp in layer_positions]

    for group, group_rois in roi_groups.items():
        # Filter to available ROIs
        available_rois = [roi for roi in group_rois if roi in p_vals_dict]
        if not available_rois:
            continue
            
        # Create figure
        fig, axes = plt.subplots(1, n_layers,
                                figsize=(n_layers * figsize_per_panel, figsize_per_panel))
        if n_layers == 1:
            axes = [axes]
        
        # Plot each ROI
        for roi_name in available_rois:
            roi_display_name, color = all_roi_colors[roi_name]
            p_vals_list = p_vals_dict[roi_name]
            t_vals_list = t_vals_dict[roi_name]
            
            for layer_idx, (p_vals, t_vals) in enumerate(zip(p_vals_list, t_vals_list)):
                pos = to_numpy(layer_positions[layer_idx])
                t_threshold_used = get_roi_t_threshold(roi_name, t_threshold)

                mask = (p_vals.flatten() < p_threshold) & (t_vals.flatten() > t_threshold_used)
                pos_filtered = pos[mask]
                
                if len(pos_filtered) > 0:
                    scatter_roi(
                        axes[layer_idx],
                        pos_filtered,
                        color,
                        roi_display_name if layer_idx == 0 else None,
                        tile_sizes[layer_idx],
                    )

        # Format axes
        for layer_idx, ax in enumerate(axes):
            ax.axis('equal')
            ax.set_aspect('equal', 'box')
            pos = to_numpy(layer_positions[layer_idx])
            dx, dy = tile_sizes[layer_idx]
            x_min, x_max = float(np.min(pos[:, 0])), float(np.max(pos[:, 0]))
            y_min, y_max = float(np.min(pos[:, 1])), float(np.max(pos[:, 1]))
            ax.set_xlim(x_min - dx / 2, x_max + dx / 2)
            ax.set_ylim(y_min - dy / 2, y_max + dy / 2)
            # remove ticks
            if group in ['face', 'body']:
                ax.set_xticks([])
            elif group in ['pSTS']:
                ax.set_yticks([])
            elif group in ['place']:
                pass
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if group == "MT":
                for spine in ax.spines.values():
                    spine.set_color('gray')
                ax.tick_params(axis='both', colors='gray')
        
        plt.tight_layout()
        path = f"{store_dir}/{prefix}{group}{suffix}.png"
        savefig(path, dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Saved {group} ROI visualization to {store_dir}")

    # make a standalone legend plot
    legend_rois = [
        'face','body','place','v6','psts','mt'
    ]
    display_names = [
        "Face","Body","Place","V6","pSTS","MT"
    ]
    fig_legend, ax_legend = plt.subplots(figsize=(4, 2))
    for roi_name, display_name in zip(legend_rois, display_names):
        roi_display_name, color = all_roi_colors[roi_name]
        ax_legend.scatter([], [], color=color, s=50, label=display_name, marker='s', edgecolors='none')
    ax_legend.legend(frameon=False, loc='center', ncol=3)
    ax_legend.axis('off')
    savefig(f"{store_dir}/{prefix}legend{suffix}.svg", dpi=dpi, bbox_inches='tight')


def _exemplar_ckpt(group_name, ckpt_list):
    if group_name == "TopoTransform":
        return MODEL_CKPT
    return ckpt_list[0]


def main():
    base_store_dir = PLOTS_DIR / "localizers"
    ensure_dir(base_store_dir)

    for group_name, ckpt_list in CKPT_GROUPS.items():
        if not ckpt_list:
            continue
        exemplar_ckpt = _exemplar_ckpt(group_name, ckpt_list)
        group_store_dir = base_store_dir / group_name.lower()
        ensure_dir(group_store_dir)
        t_vals_dict, p_vals_dict, layer_positions = localizers(exemplar_ckpt, ret_merged=True)
        plot_all_rois(
            t_vals_dict,
            p_vals_dict,
            layer_positions,
            group_store_dir,
        )


if __name__ == "__main__":
    main()
