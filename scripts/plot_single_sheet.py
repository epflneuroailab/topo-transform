"""
3D visualization of topological layer positions.

Visualizes neural network layer positions in 3D space similar to cortical tissue organization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import CACHE_DIR, PLOTS_DIR
from spacetorch.models.positions import LayerPositions

from .plot_utils import ensure_dir, savefig, to_numpy


POSITION_DIR = CACHE_DIR / "positions"


def load_layer_positions(layer_config_dir):
    """Load layer positions for a given model configuration."""
    
    layer_positions = []
    layer_config_dir = Path(layer_config_dir)
    position_files = sorted(layer_config_dir.glob("*.pkl"))
    
    for file_path in position_files:
        layer_position = LayerPositions.load(file_path)
        layer_positions.append(layer_position)
    
    return layer_positions


def visualize_3d_layers(layer_positions, layer_indices=None, 
                        figsize=(20, 5), elevation=8, azimuth=-90,
                        show_neurons=True, neuron_size=3, alpha=0.8,
                        colormap='viridis', depth_spacing=10,
                        subsample_factor=0.01,
                        color_dots=True,
                        layer_names=None, save_path=None,
                        show_shadows=True, shadow_alpha=0.9):
    
    if layer_indices is None:
        layer_indices = range(len(layer_positions))
    
    n_layers = len(layer_indices)
    n_layers = 1
    
    # Create subplots
    fig = plt.figure(figsize=figsize)
    
    for plot_idx, layer_idx in enumerate(layer_indices[:n_layers]):
        ax = fig.add_subplot(1, n_layers, plot_idx + 1, projection='3d')
        
        layer_pos = layer_positions[layer_idx]
        coords = layer_pos.coordinates
        dims = layer_pos.dims
        assert dims == (1024*3, 14, 14)

        # assume single layer
        # color the bottom-left channel of the first layer
        mask1 = np.zeros(dims, dtype=bool)
        mask1[:1024, 10:13, 0:3] = True
        mask1 = mask1.flatten()

        mask2 = np.zeros(dims, dtype=bool)
        mask2[1024:2048, 7:10, 2:5] = True
        mask2 = mask2.flatten()

        # Subsample coordinates for visualization
        if subsample_factor < 1.0:
            num_neurons = coords.shape[0]
            subsample_size = max(1, int(num_neurons * subsample_factor))
            selected_indices = np.random.choice(num_neurons, subsample_size, replace=False)
            coords = coords[selected_indices]
            mask1 = mask1[selected_indices]
            mask2 = mask2[selected_indices]
        
        # Convert to numpy if tensor
        coords = to_numpy(coords)
        
        # Add depth dimension (z-axis) with spacing between layers
        z_offset = plot_idx * depth_spacing
        z_coords = np.full(coords.shape[0], z_offset)
        
        # Get z limits for shadow projection
        z_min = 0  # Project shadows to the bottom plane
        
        if show_neurons:
            # Plot shadows first (on the bottom plane)
            if show_shadows:
                ax.scatter(coords[:, 0], coords[:, 1], 
                          z_coords - 0.06,
                          c='black', s=neuron_size * 1.6, alpha=shadow_alpha,
                          edgecolors='none')
                
            # Plot all neurons in grey with optional highlighting
            color = np.array(['#808080'] * coords.shape[0])
            sizes = np.full(coords.shape[0], neuron_size)

            # if color_dots:
            #     for i in range(coords.shape[0]):
            #         if mask1[i]:
            #             color[i] = "#6392E6"  # Highlighted color
            #             sizes[i] = neuron_size * 1.5
            #         if mask2[i]:
            #             color[i] = "#3AB8E6"  # Highlighted color
            #             sizes[i] = neuron_size * 1.5

            ax.scatter(coords[:, 0], coords[:, 1], z_coords,
                    c=color,  # Use 'c' instead of 'color' for array of colors
                    s=sizes, 
                    alpha=0.9,
                    edgecolors='white', 
                    linewidths=0.1,
                    depthshade=True)  # Enable depth shading
            
        # Set viewing angle: azimuth=0 makes x-axis parallel to screen
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Remove all axes, labels, ticks, and grid
        ax.set_axis_off()
        
        # Set aspect ratio
        ax.set_box_aspect([1, 1, 0.3])
        
        # Set limits to zoom in closer (reduce margin around data)
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        x_center = coords[:, 0].mean()
        y_center = coords[:, 1].mean()
        
        # Zoom in by using tighter limits (0.55 gives ~10% margin on each side)
        zoom_factor = 0.55
        ax.set_xlim(x_center - x_range * zoom_factor, x_center + x_range * zoom_factor)
        ax.set_ylim(y_center - y_range * zoom_factor, y_center + y_range * zoom_factor)
        ax.set_zlim(z_min - depth_spacing * 0.1, z_offset + depth_spacing * 0.1)
    
    plt.tight_layout(pad=0)
    
    if save_path:
        savefig(save_path, dpi=1200, bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"Figure saved to: {save_path}")
    
    plt.close()

    if not color_dots:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, n_layers, plot_idx + 1, projection='3d')

        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)

        # Remove axes
        ax.set_axis_off()

        # Set aspect ratio
        ax.set_box_aspect([1, 1, 0.3])

        # Add a wireframe grid above the data (no scatter points)
        xs = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 10)
        ys = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 10)
        xs, ys = np.meshgrid(xs, ys)
        zs = np.full(xs.shape, z_coords.max() + 0.1)  # slightly above the highest Z
        ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.9, linewidth=0.4)  # increase alpha & linewidth

        # Compute centers and ranges
        x_center = coords[:, 0].mean()
        y_center = coords[:, 1].mean()
        x_range = coords[:, 0].ptp() / 2
        y_range = coords[:, 1].ptp() / 2
        z_min, z_max = z_coords.min(), z_coords.max()
        depth_spacing = z_max - z_min

        # Zoom in to focus on data region
        zoom_factor = 0.55
        ax.set_xlim(x_center - x_range * zoom_factor, x_center + x_range * zoom_factor)
        ax.set_ylim(y_center - y_range * zoom_factor, y_center + y_range * zoom_factor)
        ax.set_zlim(z_min - depth_spacing * 0.1, z_max + depth_spacing * 0.2)

        if save_path:
            save_path = save_path.parent / (save_path.stem + "_grid.png")
            savefig(save_path, dpi=400, transparent=True, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

    return fig


if __name__ == "__main__":
    
    layer_position_dir = "/mnt/scratch/ytang/tdann/cache/positions/vjepa_14_18_22_single_neighbInf_sd42"

    # Create save directory if specified
    save_dir = ensure_dir(PLOTS_DIR / "plot_single_sheet")
    
    # Load layer positions
    print(f"Loading layer positions from directory: {layer_position_dir}")
    layer_positions = load_layer_positions(layer_position_dir)
    print(f"Loaded {len(layer_positions)} layers")
    
    visualize_3d_layers(
        layer_positions,
        layer_indices=None,
        save_path=save_dir / "single_sheet.png" if save_dir else None
    )

    visualize_3d_layers(
        layer_positions,
        layer_indices=None,
        azimuth=-125,
        elevation=12,
        color_dots=False,
        save_path=save_dir / "single_sheet2.png" if save_dir else None
    )
