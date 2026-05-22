import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

from .get_validate_features import validate_features
from .common import MODEL_CKPT
from config import PLOTS_DIR


# Create custom colormap from gray -> white -> red
colors = ["gray", "white", "red"]
cmap = LinearSegmentedColormap.from_list("gray_white_red", colors, N=256)


def visualize_random_autocorr(layer_features, layer_positions, dir_path, num_probes=5, suffix='', seed=None):
    # layer_features: L [B, (T), C, H, W]
    os.makedirs(dir_path, exist_ok=True)
    L = len(layer_features)
    
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    autocorrs_ = []
    probe_indices_ = []
    for l in range(L):
        lf = layer_features[l]
        
        # Handle variable dimensions [B, C, H, W] or [B, T, C, H, W]
        if lf.ndim == 5:  # [B, T, C, H, W]
            B, T, C, H, W = lf.shape
            lf = lf.reshape(-1, C, H, W)
        else:  # [B, C, H, W]
            B, C, H, W = lf.shape

        print(lf.shape)
        
        # Flatten spatial dimensions
        feats = lf.reshape(len(lf), -1).cpu().numpy()  # [B*T, N]
        pos = layer_positions[l].coordinates.cpu().numpy()  # [N, 2]
        assert pos.shape[0] == feats.shape[1], f"Position and feature size mismatch at layer {l}: {pos.shape[0]} vs {feats.shape[1]}"
        
        # Normalize features
        feats_mean = feats.mean(axis=0, keepdims=True)
        feats_std = feats.std(axis=0, keepdims=True) + 1e-9
        x = (feats - feats_mean) / feats_std
        
        # Compute autocorrelation for random probe units
        num_probes = min(num_probes, C * H * W)
        probe_indices = np.random.choice(C * H * W, num_probes, replace=False)
        probe_x = x[:, probe_indices]  # [B*T, num_probes]
        autocorrs = np.dot(probe_x.T, x) / x.shape[0]  # [num_probes, N]

        # print the top 10 correlated units for each probe
        for i, probe_idx in enumerate(probe_indices):
            autocorr = autocorrs[i]
            top10_indices = np.argsort(-autocorr)[:10]
            top10_values = autocorr[top10_indices]
            print(f"Layer {l} Probe {probe_idx} Top 10 correlated units:")
            for idx, val in zip(top10_indices, top10_values):
                print(f"  Unit {idx}: {val:.4f}")

        # Set self autocorr to 0 (ignore)
        for i, probe_idx in enumerate(probe_indices):
            autocorrs[i, probe_idx] = 0.0

        autocorrs_.append(autocorrs)
        probe_indices_.append(probe_indices)

    absvmax = max(np.max(np.abs(ac)) for ac in autocorrs_)

    for l in range(L):
        autocorrs = autocorrs_[l]  # [num_probes, N]
        probe_indices = probe_indices_[l]

        # Visualize
        fig, axes = plt.subplots(1, num_probes, figsize=(5*num_probes, 4))
        if num_probes == 1:
            axes = [axes]
        
        for i, probe_idx in enumerate(probe_indices):
            probe_pos = pos[probe_idx]  # [2]
            autocorr = autocorrs[i]  # [N]
            ax = axes[i]
            
            # Use custom colormap with adjusted range to show more variation
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=autocorr, cmap='RdBu_r', 
                          s=3, vmin=-absvmax, vmax=absvmax, alpha=1, edgecolors='none', rasterized=True) # EDITED
            ax.scatter(probe_pos[0], probe_pos[1], c='gold', s=100, 
                      marker='*', linewidths=2, edgecolors='black', zorder=10, rasterized=True) # EDITED
            
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.tick_params(axis='x', labelrotation=90) # ADDED
            ax.tick_params(axis='y', labelrotation=90) # ADDED
            ax.set_title(f'Probe {probe_idx}\n(x={probe_pos[0]:.2f}, y={probe_pos[1]:.2f})', 
                        fontsize=11, pad=10)
            
            ax.set_aspect('equal')
            ax.margins(0) # ADDED
            ax.set_xlim(pos[:, 0].min(), pos[:, 0].max()) # ADDED
            ax.set_ylim(pos[:, 1].min(), pos[:, 1].max()) # ADDED

            ax.set_facecolor('#f8f8f8')
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
        # add overall colorbar
        cbar = plt.colorbar(sc, ax=axes, fraction=0.046, pad=0.4)
        cbar.set_label('Correlation', rotation=270, labelpad=15)

        plt.suptitle(f'Layer {l} - Spatial Autocorrelations', fontsize=16, 
                    fontweight='bold', y=1.02)
        plt.tight_layout()
        # TODO adjust label sizes (right and left)
        
        # Add suffix to filename if provided
        filename = f'layer_{l}_autocorr{suffix}.svg' if suffix else f'layer_{l}_autocorr.svg'
        # filename = f'layer_{l}_autocorr{suffix}.png' if suffix else f'layer_{l}_autocorr.png'
        # plt.savefig(dir_path / filename, bbox_inches='tight', dpi=200)
        plt.savefig(dir_path / filename, bbox_inches='tight', dpi=400, format='svg')
        plt.close(fig)

        
def visualize_unit_activations_over_time(layer_features, layer_positions, dir_path, 
                                          layer_idx=-1, suffix='', seed=None, max_num_stimuli=5):
    """
    Visualize activations of 3 selected units over time and stimuli.
    
    Args:
        layer_features: List of layer features [B, T, C, H, W] or [B, C, H, W]
        layer_positions: List of position objects with .coordinates attribute
        dir_path: Directory to save the plot
        layer_idx: Which layer to visualize (default: -1, last layer)
        suffix: Suffix for filename
        seed: Random seed for reproducibility
        max_num_stimuli: Maximum number of stimuli to plot (default: 5)
    """
    os.makedirs(dir_path, exist_ok=True)
    
    if seed is not None:
        np.random.seed(seed)
    
    lf = layer_features[layer_idx]
    
    # Handle variable dimensions
    if lf.ndim == 5:  # [B, T, C, H, W]
        B, T, C, H, W = lf.shape
    else:  # [B, C, H, W]
        B, C, H, W = lf.shape
        T = 1
        lf = lf.unsqueeze(1)  # Add time dimension [B, 1, C, H, W]
    
    # Limit number of stimuli
    B = min(B, max_num_stimuli)
    lf = lf[:B]
    
    # Flatten spatial dimensions
    feats = lf.reshape(B, T, C * H * W).cpu().numpy()  # [B, T, N]
    pos = layer_positions[layer_idx].coordinates.cpu().numpy()  # [N, 2]
    
    # Normalize features across all samples
    feats_flat = feats.reshape(-1, C * H * W)  # [B*T, N]
    feats_mean = feats_flat.mean(axis=0, keepdims=True)
    feats_std = feats_flat.std(axis=0, keepdims=True) + 1e-9
    x = (feats_flat - feats_mean) / feats_std
    
    # Compute autocorrelation to find clusters
    # Pick a unit in the rightmost area as reference
    rightmost_units = np.argsort(pos[:, 0])[-int(0.2 * len(pos)):]  # Top 20% rightmost
    probe_idx = np.random.choice(rightmost_units)
    probe_x = x[:, probe_idx:probe_idx+1]  # [B*T, 1]
    autocorr = np.dot(probe_x.T, x).flatten() / x.shape[0]  # [N]
    
    # Find highly correlated units in the rightmost region (red cluster)
    red_cluster_mask = (autocorr > 0.7) & (pos[:, 0] > np.percentile(pos[:, 0], 50))
    red_cluster_indices = np.where(red_cluster_mask)[0]
    
    # Select 2 units close together in the red cluster
    if len(red_cluster_indices) >= 2:
        # Find two close units
        idx1 = np.random.choice(red_cluster_indices)
        distances = np.linalg.norm(pos[red_cluster_indices] - pos[idx1], axis=1)
        close_mask = distances > 0  # Exclude the same unit
        if np.any(close_mask):
            close_indices = red_cluster_indices[close_mask]
            distances_nonzero = distances[close_mask]
            idx2 = close_indices[np.argmin(distances_nonzero)]
        else:
            print("Warning: No close units found in red cluster, picking another unit.")
            idx2 = red_cluster_indices[0] if red_cluster_indices[0] != idx1 else red_cluster_indices[-1]
    else:
        # Fallback: just pick rightmost units
        print("Warning: Not enough units in red cluster, picking rightmost units.")
        idx1, idx2 = rightmost_units[-2], rightmost_units[-1]
    
    # Find an anticorrelated unit (far from the cluster)
    anticorr_mask = autocorr < 0.0
    if np.any(anticorr_mask):
        anticorr_indices = np.where(anticorr_mask)[0]
        idx3 = np.random.choice(anticorr_indices)
    else:
        # Fallback: pick a leftmost unit
        print("Warning: No anticorrelated units found, picking leftmost unit.")
        idx3 = np.argsort(pos[:, 0])[0]
    
    selected_indices = [idx1, idx2, idx3]
    selected_positions = pos[selected_indices]
    selected_activations = feats[:, :, selected_indices]  # [B, T, 3]
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 3], hspace=0.3, wspace=0.3)
    
    # Left column: Show spatial location of selected units
    ax_spatial = fig.add_subplot(gs[:, 0])

    absvmax = np.max(np.abs(autocorr))
    sc = ax_spatial.scatter(pos[:, 0], pos[:, 1], c=autocorr, cmap='RdBu_r',
                           s=1.5, vmin=-absvmax, vmax=absvmax, alpha=1, edgecolors='none')
    
    # Mark selected units
    unit_colors = ['red', 'red', 'red']
    unit_labels = ['Unit 1 (cluster)', 'Unit 2 (cluster)', 'Unit 3 (distant)']
    for i, (idx, color, label) in enumerate(zip(selected_indices, unit_colors, unit_labels)):
        ax_spatial.scatter(pos[idx, 0], pos[idx, 1], c=color, s=200,
                          marker='o', linewidths=3, edgecolors='black', 
                          zorder=10, label=label)
    
    ax_spatial.set_xlabel('X Position', fontsize=12)
    ax_spatial.set_ylabel('Y Position', fontsize=12)
    ax_spatial.set_title('Selected Units\n(Autocorrelation Map)', fontsize=13, fontweight='bold')
    ax_spatial.set_aspect('equal')
    ax_spatial.set_facecolor('#f8f8f8')
    ax_spatial.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    # ax_spatial.legend(loc='upper right', fontsize=9)
    plt.colorbar(sc, ax=ax_spatial, fraction=0.046, pad=0.04, label='Correlation')
    
    # Right column: Plot activations over time for each unit
    time_axis = np.arange(B * T)
    
    for unit_idx, (idx, color, label) in enumerate(zip(selected_indices, unit_colors, unit_labels)):
        ax = fig.add_subplot(gs[unit_idx, 1])
        
        activations = selected_activations[:, :, unit_idx].flatten()  # [B*T]
        
        ax.plot(time_axis, activations, color=color, linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add vertical dashed lines between stimuli (corrected position)
        for b in range(1, B):
            ax.axvline(x=b * T - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_ylabel('Activation', fontsize=11)
        # ax.set_title(f'{label}\nPosition: ({pos[idx, 0]:.2f}, {pos[idx, 1]:.2f})', 
        #             fontsize=11, fontweight='bold')
        print(f'{label}\nPosition: ({pos[idx, 0]:.2f}, {pos[idx, 1]:.2f})')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        
        if unit_idx == 2:  # Last row
            ax.set_xlabel('Time (across stimuli)', fontsize=11)
        else:
            ax.set_xticklabels([])
    
    fig.suptitle(f'Layer {layer_idx} - Unit Activations Over Time and Stimuli (First {B} Stimuli)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    filename = f'layer_{layer_idx}_unit_activations{suffix}.png' if suffix else f'layer_{layer_idx}_unit_activations.png'
    plt.savefig(dir_path / filename, bbox_inches='tight', dpi=400, format='png')
    plt.close(fig)


if __name__ == '__main__':
    all_features, positions = validate_features(MODEL_CKPT)
    viz_dir = PLOTS_DIR / 'plot_autocorr'
    viz_dir.mkdir(parents=True, exist_ok=True)

    seed = 40 # For reproducibility
    visualize_random_autocorr(all_features, positions, viz_dir, seed=seed)
    visualize_unit_activations_over_time(all_features, positions, viz_dir, layer_idx=0, seed=seed)
