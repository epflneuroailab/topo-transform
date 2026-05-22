import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def _as_numpy_positions(positions):
    if isinstance(positions, torch.Tensor):
        return positions.cpu().numpy()
    return positions


def _save_figure(path, dpi):
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()


def _fill_nan_with_zero(t_vals_dict):
    t_vals_dict_filled = {}
    for cat_name, t_vals_list in t_vals_dict.items():
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        t_vals_filled = [np.nan_to_num(t_vals_layer, nan=0.0) for t_vals_layer in t_vals_list]
        t_vals_dict_filled[cat_name] = t_vals_filled
    return t_vals_dict_filled

def visualize_tvals(t_vals_dict, layer_positions, store_dir, figsize_per_panel=5, prefix='', suffix='', 
                    topk_percent=100, dpi=150):
    """Visualize t-statistics for each category and layer.
    
    Args:
        t_vals_dict: Dict mapping category names to list of t-value arrays (one per layer)
        layer_positions: List of position arrays for each layer [num_units, 2]
        store_dir: Directory to save visualizations
        figsize_per_panel: Size of each subplot panel
        topk_percent: float, percentage of units to highlight (default: 100)
    """
    os.makedirs(store_dir, exist_ok=True)
    
    # make a copy
    t_vals_dict = _fill_nan_with_zero(t_vals_dict)
    
    categories = list(t_vals_dict.keys())
    n_layers = len(t_vals_dict[categories[0]]) if categories else 0
    
    # Create separate figure for each category
    for cat_name, t_vals_list in t_vals_dict.items():
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        
        n_cols = len(t_vals_list)
        fig, axes = plt.subplots(1, n_cols, 
                                figsize=(n_cols * figsize_per_panel, figsize_per_panel))
        
        # Ensure axes is always a list
        if n_cols == 1:
            axes = [axes]
        
        # Compute global normalization across all layers for this category
        all_t_vals = np.concatenate([t_vals.flatten() for t_vals in t_vals_list])
        vmax_global = np.abs(all_t_vals).max()
        norm_global = Normalize(vmin=-vmax_global, vmax=vmax_global)
        
        # Compute global threshold for topk_percent across all layers
        if topk_percent < 100:
            k_global = int(len(all_t_vals) * topk_percent / 100)
            threshold_global = np.partition(all_t_vals, -k_global)[-k_global]
        else:
            threshold_global = -np.inf
        
        for layer_idx, t_vals in enumerate(t_vals_list):
            pos = _as_numpy_positions(layer_positions[layer_idx])
            
            # Apply topk_percent filtering using global threshold
            t_vals_flat = t_vals.flatten()
            if topk_percent < 100:
                mask = t_vals_flat >= threshold_global
                pos_filtered = pos[mask]
                t_vals_filtered = t_vals_flat[mask]
            else:
                pos_filtered = pos
                t_vals_filtered = t_vals_flat
            
            # Create scatter plot with global normalization
            im = axes[layer_idx].scatter(pos_filtered[:, 0], pos_filtered[:, 1], 
                                        c=t_vals_filtered, 
                                        cmap='bwr', norm=norm_global, s=0.1)
            title = f'Layer {layer_idx}'
            if topk_percent < 100:
                title += f' (top {topk_percent}%)'
            axes[layer_idx].set_title(title)
            axes[layer_idx].axis('equal')
            axes[layer_idx].set_aspect('equal', 'box')
            
        # Add colorbar
        cbar_ax = fig.add_axes([0.375, -0.07, 0.25, 0.03])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='t-statistic')
        plt.suptitle(f'{cat_name} (one-vs-rest)', fontsize=14, y=1.02)
        plt.tight_layout()
        _save_figure(f'{store_dir}/{prefix}tvals_{cat_name}{suffix}.png', dpi)
    
    print(f"Saved visualizations to {store_dir}")


def visualize_all_rois(t_vals_dicts, layer_positions, store_dir, figsize_per_panel=5, 
                       prefix='', suffix='', topk_percent=100, dpi=150, alpha=0.6):
    """Visualize all ROIs (categories) together across layers using different colors.
    
    Args:
        t_vals_dicts: List of dicts or single dict. Each dict maps ROI/category names to 
                     list of t-value arrays (one per layer)
        layer_positions: List of position arrays for each layer [num_units, 2]
        store_dir: Directory to save visualizations
        figsize_per_panel: Size of each subplot panel
        topk_percent: float, percentage of units to highlight per ROI (default: 100)
        alpha: Transparency for overlapping points
    """
    os.makedirs(store_dir, exist_ok=True)
    
    # Handle both single dict and list of dicts
    if isinstance(t_vals_dicts, dict):
        t_vals_dicts = [t_vals_dicts]
    
    t_vals_dicts = [_fill_nan_with_zero(t_vals_dict) for t_vals_dict in t_vals_dicts]

    # Collect all ROI names (categories) across all dicts
    all_rois = []
    for t_vals_dict in t_vals_dicts:
        all_rois.extend(list(t_vals_dict.keys()))
    unique_rois = sorted(set(all_rois))
    
    # Determine number of layers
    first_dict = t_vals_dicts[0]
    first_roi = list(first_dict.keys())[0]
    t_vals_list = first_dict[first_roi]
    if not isinstance(t_vals_list, list):
        t_vals_list = [t_vals_list]
    n_layers = len(t_vals_list)
    
    # Create color map for different ROIs
    n_rois = len(unique_rois)
    roi_colors = plt.cm.tab20(np.linspace(0, 1, n_rois))
    roi_to_color = {roi: roi_colors[i] for i, roi in enumerate(unique_rois)}
    
    # Create combined view showing all ROIs together
    fig, axes = plt.subplots(1, n_layers,
                            figsize=(n_layers * figsize_per_panel, figsize_per_panel))
    if n_layers == 1:
        axes = [axes]
    
    # Track which ROIs we've added to legend
    legend_added = set()
    
    for t_vals_dict in t_vals_dicts:
        for roi_name in t_vals_dict.keys():
            t_vals_list = t_vals_dict[roi_name]
            if not isinstance(t_vals_list, list):
                t_vals_list = [t_vals_list]
            
            # Compute threshold for this ROI
            all_t_vals = np.concatenate([t_vals.flatten() for t_vals in t_vals_list])
            if topk_percent < 100:
                k = int(len(all_t_vals) * topk_percent / 100)
                threshold = np.partition(all_t_vals, -k)[-k]
            else:
                threshold = -np.inf
            
            for layer_idx, t_vals in enumerate(t_vals_list):
                pos = _as_numpy_positions(layer_positions[layer_idx])
                
                # Filter by threshold
                t_vals_flat = t_vals.flatten()
                if topk_percent < 100:
                    mask = t_vals_flat >= threshold
                    pos_filtered = pos[mask]
                    t_vals_filtered = t_vals_flat[mask]
                else:
                    pos_filtered = pos
                    t_vals_filtered = t_vals_flat
                
                # Normalize t-values to [0, 1] for alpha intensity
                if len(t_vals_filtered) > 0:
                    t_norm = (t_vals_filtered - t_vals_filtered.min()) / \
                            (t_vals_filtered.max() - t_vals_filtered.min() + 1e-8)
                    
                    # Plot with ROI-specific color and t-value intensity
                    # Only add label for first layer and first occurrence of this ROI
                    add_label = (layer_idx == 0 and roi_name not in legend_added)
                    
                    axes[layer_idx].scatter(
                        pos_filtered[:, 0], 
                        pos_filtered[:, 1],
                        c=[roi_to_color[roi_name]], 
                        s=0.5,
                        alpha=alpha * (0.3 + 0.7 * t_norm),  # Scale alpha by t-value
                        label=roi_name if add_label else None
                    )
                    
                    if add_label:
                        legend_added.add(roi_name)
    
    # Format axes
    for layer_idx, ax in enumerate(axes):
        title = f'Layer {layer_idx}'
        if topk_percent < 100:
            title += f' (top {topk_percent}%)'
        ax.set_title(title)
        ax.axis('equal')
        ax.set_aspect('equal', 'box')
    
    # Add legend
    if legend_added:
        axes[0].legend(loc='upper left', markerscale=10, framealpha=0.9, 
                      fontsize=8, ncol=max(1, len(unique_rois) // 8))
    
    plt.suptitle('All ROIs Combined', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_figure(f'{store_dir}/{prefix}all_rois_combined{suffix}.png', dpi)
    
    print(f"Saved combined ROI visualization to {store_dir}")


def visualize_all_rois_v2(t_vals_dicts, layer_positions, store_dir, figsize_per_panel=5, 
                       prefix='', suffix='', topk_percent=1, dpi=250):
    """Visualize all ROIs (categories) together across layers using different colors.
    
    Args:
        t_vals_dicts: List of dicts or single dict. Each dict maps ROI/category names to 
                     list of t-value arrays (one per layer)
        layer_positions: List of position arrays for each layer [num_units, 2]
        store_dir: Directory to save visualizations
        figsize_per_panel: Size of each subplot panel
        topk_percent: float, percentage of units to highlight per ROI (default: 100)
    """
    os.makedirs(store_dir, exist_ok=True)

    # Handle both single dict and list of dicts
    if isinstance(t_vals_dicts, dict):
        t_vals_dicts = [t_vals_dicts]

    t_vals_dicts = [_fill_nan_with_zero(t_vals_dict) for t_vals_dict in t_vals_dicts]

    all_roi_colors = {
        # static categorical ROIs - saturated, bold colors
        "face": ("static-face", (0.75, 0.00, 0.00)),        # crimson
        "body": ("static-body", (0.00, 0.45, 0.00)),        # dark green
        "object": ("static-object", (0.00, 0.20, 0.65)),    # navy blue
        "place": ("static-place", (0.80, 0.35, 0.00)),      # burnt orange
        "character": ("static-word", (0.40, 0.00, 0.60)),   # dark violet

        # dynamic categorical ROIs - pastel/desaturated versions with strong contrast
        "Faces": ("dynamic-face", (1.00, 0.80, 0.80)),      # blush pink
        "Bodies": ("dynamic-body", (0.80, 1.00, 0.80)),     # pale mint
        "Objects": ("dynamic-object", (0.75, 0.85, 1.00)),  # sky blue
        "Scenes": ("dynamic-place", (1.00, 0.90, 0.70)),    # light peach
        
        # dynamic motion ROIs - cyan/teal family (more distinct)
        "V6": ("V6", (0.0, 0.85, 0.95)),                  # bright cyan (cool)
        "MT": ("MT", (0.95, 0.20, 0.70)),                 # vibrant magenta (warm)
        "pSTS": ("pSTS", (0.85, 0.85, 0.0)),              # bright yellow (neutral)

        # placeholder for motion-vs-static contrast
        "motion_vs_static": ("motion_vs_static", (0.0, 0.0, 0.0)),
        # "motion_vs_static_v2": ("motion_vs_static", (0.0, 0.0, 0.0)),
        # "motion_vs_static_v3": ("motion_vs_static", (0.0, 0.0, 0.0)),
    }

    # motion_vs_statics = ["motion_vs_static", "motion_vs_static_v2", "motion_vs_static_v3"]
    motion_vs_statics = ["motion_vs_static"]

    for group, group_rois in {
        "static-categorical": ["face", "body", "object", "place", "character", *motion_vs_statics],
        "dynamic-categorical": ["Faces", "Bodies", "Objects", "Scenes", *motion_vs_statics],
        "dynamic-motion": ["V6", "MT", "pSTS", *motion_vs_statics],
        "face": ["face", "Faces", *motion_vs_statics],
        "body": ["body", "Bodies", *motion_vs_statics],
        "object": ["object", "Objects", *motion_vs_statics],
        "place": ["place", "Scenes", *motion_vs_statics],
    }.items():
        roi_colors = {roi: all_roi_colors[roi] for roi in group_rois}
        
        # Combine all dicts
        combined_t_vals_dict = {}
        for d in t_vals_dicts:
            combined_t_vals_dict.update(d)
        
        # Filter to only ROIs in roi_colors
        combined_t_vals_dict = {roi_name: combined_t_vals_dict[roi_name] 
                            for roi_name in roi_colors.keys() 
                            if roi_name in combined_t_vals_dict}
        
        # Separate categorical ROIs from motion ROIs
        categorical_rois = {k: v for k, v in combined_t_vals_dict.items() 
                        if not k.startswith("motion_vs_static")}
        motion_rois = {k: v for k, v in combined_t_vals_dict.items() 
                    if k.startswith("motion_vs_static")}
        
        # Determine number of layers
        first_roi = list(combined_t_vals_dict.keys())[0]
        t_vals_list = combined_t_vals_dict[first_roi]
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        n_layers = len(t_vals_list)
        
        # Calculate number of rows: 1 for categorical + 1 per motion ROI
        n_rows = 1 + len(motion_rois)
        
        # Create figure with appropriate number of rows
        fig, axes = plt.subplots(n_rows, n_layers,
                                figsize=(n_layers * figsize_per_panel, n_rows * figsize_per_panel))
        if n_layers == 1:
            axes = axes.reshape(n_rows, 1)
        elif n_rows == 1:
            axes = axes.reshape(1, n_layers)
        
        # Track which ROIs we've added to legend
        legend_added = set()

        # Plot categorical ROIs in first row
        for roi_name in categorical_rois.keys():
            t_vals_list = categorical_rois[roi_name]
            if not isinstance(t_vals_list, list):
                t_vals_list = [t_vals_list]
            
            # Compute threshold for this ROI
            all_t_vals = np.concatenate([t_vals.flatten() for t_vals in t_vals_list])
            if topk_percent < 100:
                k = int(len(all_t_vals) * topk_percent / 100)
                threshold = np.partition(all_t_vals, -k)[-k]
            else:
                threshold = -np.inf
            
            for layer_idx, t_vals in enumerate(t_vals_list):
                pos = layer_positions[layer_idx]
                if isinstance(pos, torch.Tensor):
                    pos = pos.cpu().numpy()
                
                # Filter by threshold
                t_vals_flat = t_vals.flatten()
                if topk_percent < 100:
                    mask = t_vals_flat >= threshold
                    pos_filtered = pos[mask]
                    t_vals_filtered = t_vals_flat[mask]
                else:
                    pos_filtered = pos
                    t_vals_filtered = t_vals_flat
                
                # Normalize t-values to [0, 1] for alpha intensity
                if len(t_vals_filtered) > 0:
                    t_norm = (t_vals_filtered - t_vals_filtered.min()) / \
                            (t_vals_filtered.max() - t_vals_filtered.min() + 1e-8)

                    # Only add label for first layer and first occurrence of this ROI
                    add_label = (layer_idx == 0 and roi_name not in legend_added)
                    
                    roi_display_name, color = roi_colors[roi_name]

                    axes[0, layer_idx].scatter(
                        pos_filtered[:, 0], 
                        pos_filtered[:, 1],
                        color=color, 
                        s=0.5,
                        # alpha=(0.3 + 0.7 * t_norm),  # Scale alpha by t-value
                        alpha=0.8,
                        label=roi_display_name if add_label else None
                    )
                    
                    if add_label:
                        legend_added.add(roi_display_name)
        
        # Format axes for categorical ROIs row
        for layer_idx in range(n_layers):
            title = f'Layer {layer_idx}'
            if topk_percent < 100:
                title += f' (top {topk_percent}%)'
            axes[0, layer_idx].set_title(title)
            axes[0, layer_idx].axis('equal')
            axes[0, layer_idx].set_aspect('equal', 'box')
        
        # Add legend to categorical row
        if legend_added:
            axes[0, 0].legend(loc='upper left', markerscale=10, framealpha=1, 
                        fontsize=8, ncol=max(1, len(categorical_rois) // 8))
        
        # Plot each motion ROI in its own row
        for motion_idx, roi_name in enumerate(motion_rois.keys(), start=1):
            t_vals_list = motion_rois[roi_name]
            if not isinstance(t_vals_list, list):
                t_vals_list = [t_vals_list]
            
            for layer_idx, t_vals in enumerate(t_vals_list):
                pos = layer_positions[layer_idx]
                if isinstance(pos, torch.Tensor):
                    pos = pos.cpu().numpy()
                
                t_vals_flat = t_vals.flatten()
                
                # Plot motion contrast with contour plot
                if len(t_vals_flat) > 0:
                    axes[motion_idx, layer_idx].tricontourf(
                        pos[:, 0], 
                        pos[:, 1], 
                        -t_vals_flat,
                        levels=50,
                        cmap='Spectral',
                        alpha=1
                    )
                
                # Format axes
                axes[motion_idx, layer_idx].set_title(f'{roi_name} - Layer {layer_idx}')
                axes[motion_idx, layer_idx].axis('equal')
                axes[motion_idx, layer_idx].set_aspect('equal', 'box')
        
        plt.suptitle('All ROIs Combined', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{store_dir}/{group}_{prefix}all_rois_combined{suffix}.png', 
                    dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined ROI {group} visualization to {store_dir}")
