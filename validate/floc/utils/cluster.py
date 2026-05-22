import numpy as np
import skimage.measure
import shapely.geometry
import matplotlib.pyplot as plt
import matplotlib.patches
from typing import List, Optional
import torch
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist
from spacetorch.utils.spatial_utils import concave_hull


class Patch:
    """Represents a category-selective patch."""
    
    def __init__(
        self,
        positions: np.ndarray,
        unit_indices: np.ndarray,
        selectivities: np.ndarray,
        p_values: Optional[np.ndarray] = None,
        hull_alpha: float = 0.1,
    ):
        self.positions = positions
        self.unit_indices = unit_indices
        self.selectivities = selectivities
        self.p_values = p_values
        
        self._points = [shapely.geometry.Point(p) for p in self.positions]
        self.concave_hull = concave_hull(self._points, alpha=hull_alpha)
    
    @property
    def center(self) -> np.ndarray:
        return np.array(self.concave_hull.centroid.coords)[0]
    
    @property
    def area(self) -> float:
        return self.concave_hull.area
    
    @property
    def hull_vertices(self) -> np.ndarray:
        return np.array(list(zip(*self.concave_hull.exterior.coords.xy)))
    
    def to_mpl_poly(self, color="red", alpha: float = 0.6, lw: float = 2, hollow: bool = False):
        edgecolor = color if hollow else "white"
        fill = not hollow
        return matplotlib.patches.Polygon(
            self.hull_vertices,
            facecolor=color,
            alpha=alpha,
            edgecolor=edgecolor,
            lw=lw,
            fill=fill,
        )


_BACKGROUND_CLUSTER = 0


def _to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array


def labels_to_unit_indices(labels, positions):
    """Convert cluster labels to unit indices."""
    unique_labels = np.unique(labels)
    index_sets = []
    
    for lab in unique_labels:
        if lab == _BACKGROUND_CLUSTER:
            continue

        unit_ids = np.where(labels == lab)[0]
        if len(unit_ids) == 0:
            continue

        matching_pos = positions[unit_ids]
        distances = cdist(positions, matching_pos)
        min_distance_for_each_unit = distances.min(axis=1)

        min_cutoff = 2**0.5 + 1e-3
        units_in_cluster = np.where(min_distance_for_each_unit <= min_cutoff)[0]

        if len(units_in_cluster) >= 3:
            index_sets.append(units_in_cluster)

    return index_sets


def _to_regular_grid(positions: np.ndarray, values: np.ndarray, tolerance: float = 1e-6):
    """Convert scattered positions and values to regular grid."""
    values = values.flatten()
    assert positions.shape[0] == values.shape[0]
    
    unique_x = np.unique(np.round(positions[:, 0] / tolerance) * tolerance)
    unique_y = np.unique(np.round(positions[:, 1] / tolerance) * tolerance)
    
    height, width = len(unique_y), len(unique_x)
    assert height * width == positions.shape[0]
    
    x_to_col = {x: i for i, x in enumerate(unique_x)}
    y_to_row = {y: i for i, y in enumerate(unique_y)}
    
    grid_values = np.full((height, width), np.nan)
    grid_positions = np.full((height, width, 2), np.nan)
    
    for pos, val in zip(positions, values):
        x_rounded = unique_x[np.argmin(np.abs(unique_x - pos[0]))]
        y_rounded = unique_y[np.argmin(np.abs(unique_y - pos[1]))]
        
        row, col = y_to_row[y_rounded], x_to_col[x_rounded]
        grid_values[row, col] = val
        grid_positions[row, col] = pos
    
    return grid_values, grid_positions


def find_patches(
    positions: np.ndarray,
    selectivities: np.ndarray,
    p_values: np.ndarray,
    t_threshold: float = 0.0,
    p_threshold: float = 0.01,
    minimum_size: float = 50,
    maximum_size: float = 4500,
    min_count: int = 10,
    hull_alpha: float = 0.1,
) -> List[Patch]:
    """Find category-selective patches."""
    positions = _to_numpy(positions)
    selectivities = _to_numpy(selectivities)
    p_values = _to_numpy(p_values)

    selectivities = selectivities.copy()
    positions = positions.copy()
    p_values = p_values.copy()

    assert selectivities.shape[0] == 1
    selectivities = selectivities[0]
    assert p_values.shape[0] == 1
    p_values = p_values[0]

    selectivities, _ = _to_regular_grid(positions, selectivities)
    p_values, positions = _to_regular_grid(positions, p_values)
    positions = positions.reshape(-1, 2)
    
    labels = skimage.measure.label((selectivities > t_threshold) & (p_values < p_threshold) & (~np.isnan(selectivities)))
    clusters = labels_to_unit_indices(labels.flatten(), positions)
    
    patches = []
    for cluster in clusters:
        point_positions = positions[cluster]

        if np.any(np.ptp(point_positions, axis=0) == 0):
            continue

        patch = Patch(
            positions=point_positions,
            unit_indices=cluster,
            selectivities=selectivities.flatten()[cluster],
            p_values=p_values.flatten()[cluster],
            hull_alpha=hull_alpha,
        )
        
        if (patch.area >= minimum_size and 
            patch.area <= maximum_size and 
            len(patch.unit_indices) >= min_count):
            patches.append(patch)

    return patches


def find_patches_for_categories(
    t_vals_dict: dict,
    p_vals_dict: dict,
    layer_positions: List[np.ndarray],
    t_threshold: float = 0.0,
    p_threshold: float = 0.01,
    minimum_size: float = 50,
    maximum_size: float = 4500,
    min_count: int = 10,
    hull_alpha: float = 0.1,
    verbose: bool = False,
) -> dict:
    """Find patches for all categories and layers."""
    results = {}
    
    pbar_cats = tqdm(zip(t_vals_dict.items(), p_vals_dict.items()), desc="Processing categories", disable=not verbose)
    
    for (cat_name, t_vals_list), (cat_name_, p_vals_list) in pbar_cats:
        assert cat_name == cat_name_, "Category names do not match between t-values and p-values."
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        
        if not isinstance(p_vals_list, list):
            p_vals_list = [p_vals_list]
        
        category_patches = []
        pbar_cats.set_postfix_str(f"Category: {cat_name}")
        
        for layer_idx in tqdm(range(len(t_vals_list)), 
                              desc=f"  Layers for {cat_name}", 
                              leave=False,
                              disable=not verbose):
            t_vals = t_vals_list[layer_idx]
            p_vals = p_vals_list[layer_idx]
            
            patches = find_patches(
                positions=layer_positions[layer_idx],
                p_values=p_vals,
                selectivities=t_vals,
                t_threshold=t_threshold,
                p_threshold=p_threshold,
                minimum_size=minimum_size,
                maximum_size=maximum_size,
                min_count=min_count,
                hull_alpha=hull_alpha,
            )
            category_patches.append(patches)
        
        results[cat_name] = category_patches
    
    return results


def visualize_patches(
    t_vals_dict: dict,
    p_vals_dict: dict,
    layer_positions: List[np.ndarray],
    viz_dir: str,
    prefix: str = '',
    suffix: str = '',
    t_threshold: float = 0.0,
    p_threshold: float = 0.01,
    minimum_size: float = 50,
    maximum_size: float = 4500,
    min_count: int = 10,
    hull_alpha: float = 0.1,
    verbose: bool = False,
    figsize_per_panel: int = 6,
    color_map: Optional[dict] = None,
):
    """Find and visualize patches (individual + joint plot only)."""
    os.makedirs(viz_dir, exist_ok=True)
    
    # Find patches
    patch_results = find_patches_for_categories(
        t_vals_dict=t_vals_dict,
        p_vals_dict=p_vals_dict,
        layer_positions=layer_positions,
        t_threshold=t_threshold,
        p_threshold=p_threshold,
        minimum_size=minimum_size,
        maximum_size=maximum_size,
        min_count=min_count,
        hull_alpha=hull_alpha,
        verbose=verbose,
    )
    
    # Setup colors
    categories = list(patch_results.keys())
    if color_map is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        color_map = {cat: colors[i] for i, cat in enumerate(categories)}
    
    # Individual plots
    print("\nCreating individual plots...")
    for cat_name, patches_by_layer in tqdm(patch_results.items(), desc="Individual plots"):
        for layer_idx, patches in enumerate(patches_by_layer):
            _plot_single(patches, layer_positions[layer_idx], cat_name, layer_idx,
                        viz_dir, color_map.get(cat_name, 'blue'), 
                        (figsize_per_panel, figsize_per_panel), prefix, suffix)
    
    # Joint plot (all categories, all layers)
    print("\nCreating joint plot...")
    _plot_joint(patch_results, layer_positions, viz_dir, color_map, 
                figsize_per_panel, prefix, suffix)
    
    print(f"\nAll visualizations saved to {viz_dir}")
    return patch_results


def _plot_single(patches, all_positions, cat_name, layer_idx, store_dir, 
                color, figsize, prefix, suffix):
    """Plot single category-layer combination."""
    all_positions = _to_numpy(all_positions)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for i, patch in enumerate(patches):
        poly = patch.to_mpl_poly(alpha=0.2, hollow=False)
        poly.set_facecolor(color)
        ax.add_patch(poly)
        ax.text(patch.center[0], patch.center[1], f"{i+1}", fontsize=10, 
               fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='circle', facecolor='white', edgecolor=color, linewidth=2))
    
    ax.set_title(f'{cat_name} - Layer {layer_idx} ({len(patches)} patches)', 
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal', 'box')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{store_dir}/{prefix}{cat_name}_layer{layer_idx}{suffix}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_joint(patch_results, layer_positions, store_dir, color_map, 
               figsize_per_panel, prefix, suffix):
    """Create joint plot showing all categories across all layers."""
    categories = list(patch_results.keys())
    n_layers = len(layer_positions)
    
    fig, axes = plt.subplots(1, n_layers, 
                            figsize=(n_layers * figsize_per_panel, figsize_per_panel))
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx, ax in enumerate(axes):
        # Plot all categories on this layer
        for cat_name in categories:
            patches = patch_results[cat_name][layer_idx]
            color = color_map.get(cat_name, 'blue')
            
            for i, patch in enumerate(patches):
                poly = patch.to_mpl_poly(alpha=0.2, hollow=False)
                poly.set_facecolor(color)
                ax.add_patch(poly)
                ax.text(patch.center[0], patch.center[1], f"{i+1}", fontsize=8,
                       fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='circle', facecolor='white', 
                                edgecolor=color, linewidth=1.5))
        
        ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', 'box')
        ax.axis('equal')
    
    # Legend
    handles = [matplotlib.patches.Patch(color=color_map.get(cat, 'blue'), label=cat) 
               for cat in categories]
    fig.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.suptitle('Joint Patch Visualization - All Categories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{store_dir}/{prefix}joint_all_categories{suffix}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


__all__ = [
    "Patch",
    "find_patches",
    "find_patches_for_categories",
    "labels_to_unit_indices",
    "visualize_patches",
]
