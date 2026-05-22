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
        torch.manual_seed(seed)

    # Determine device
    device = layer_features[0].device

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
        
        # Flatten spatial dimensions - KEEP ON GPU
        feats = lf.reshape(len(lf), -1)  # [B*T, N] - stays on GPU
        pos = layer_positions[l].coordinates  # [N, 2] - stays on GPU
        assert pos.shape[0] == feats.shape[1], f"Position and feature size mismatch at layer {l}: {pos.shape[0]} vs {feats.shape[1]}"
        
        # Normalize features - GPU accelerated
        feats_mean = feats.mean(dim=0, keepdim=True)
        feats_std = feats.std(dim=0, keepdim=True) + 1e-9
        x = (feats - feats_mean) / feats_std
        
        # Compute autocorrelation for random probe units
        num_probes_actual = min(num_probes, C * H * W)
        probe_indices = np.random.choice(C * H * W, num_probes_actual, replace=False)
        probe_x = x[:, probe_indices]  # [B*T, num_probes]
        
        # GPU-accelerated matrix multiplication
        autocorrs = torch.mm(probe_x.T, x) / x.shape[0]  # [num_probes, N]

        # Set self autocorr to 0 (ignore) - GPU operation
        for i, probe_idx in enumerate(probe_indices):
            autocorrs[i, probe_idx] = 0.0

        # Convert to CPU only when needed for plotting
        autocorrs_.append(autocorrs.cpu().numpy())
        probe_indices_.append(probe_indices)

    absvmax = max(np.max(np.abs(ac)) for ac in autocorrs_)

    # Convert positions to numpy once for all plotting
    pos_np = pos.cpu().numpy()

    for l in range(L):
        autocorrs = autocorrs_[l]  # [num_probes, N]
        probe_indices = probe_indices_[l]

        # Visualize
        fig, axes = plt.subplots(1, num_probes_actual, figsize=(5*num_probes_actual, 4))
        if num_probes_actual == 1:
            axes = [axes]
        
        for i, probe_idx in enumerate(probe_indices):
            print(f"Layer {l}, Probe {i}: Unit index {probe_idx}")
            probe_pos = pos_np[probe_idx]  # [2]
            autocorr = autocorrs[i]  # [N]
            ax = axes[i]
            
            # Use custom colormap with adjusted range to show more variation
            sc = ax.scatter(pos_np[:, 0], pos_np[:, 1], c=autocorr, cmap='RdBu_r', 
                          s=3, vmin=-absvmax, vmax=absvmax, alpha=1, edgecolors='none', rasterized=True)
            ax.scatter(probe_pos[0], probe_pos[1], c='gold', s=100, 
                      marker='*', linewidths=2, edgecolors='black', zorder=10, rasterized=True)
            
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.tick_params(axis='x', labelrotation=90)
            ax.tick_params(axis='y', labelrotation=90)
            ax.set_title(f'Probe {probe_idx}\n(x={probe_pos[0]:.2f}, y={probe_pos[1]:.2f})', 
                        fontsize=11, pad=10)
            
            ax.set_aspect('equal')
            ax.margins(0)
            ax.set_xlim(pos_np[:, 0].min(), pos_np[:, 0].max())
            ax.set_ylim(pos_np[:, 1].min(), pos_np[:, 1].max())

            ax.set_facecolor('#f8f8f8')
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
        # add overall colorbar
        cbar = plt.colorbar(sc, ax=axes, fraction=0.046, pad=0.4)
        cbar.set_label('Correlation', rotation=270, labelpad=15)

        plt.suptitle(f'Layer {l} - Spatial Autocorrelations', fontsize=16, 
                    fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Add suffix to filename if provided
        filename = f'layer_{l}_autocorr{suffix}.svg' if suffix else f'layer_{l}_autocorr.svg'
        plt.savefig(dir_path / filename, bbox_inches='tight', dpi=400, format='svg')
        plt.close(fig)

    return probe_indices
        
def visualize_unit_activations_over_time(layer_features, layer_positions, dir_path,
                                          layer_idx=-1, suffix='', seed=None, max_num_stimuli=5,
                                          probe_idx=None):
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
        probe_idx: Index of probe unit (if None, randomly selected)
    """
    os.makedirs(dir_path, exist_ok=True)
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    lf = layer_features[layer_idx]
    device = lf.device
    
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
    print("The correlation might look different from the autocorr map due to sampling variance.")
    
    # Flatten spatial dimensions - KEEP ON GPU
    feats = lf.reshape(B, T, C * H * W)  # [B, T, N]
    pos = layer_positions[layer_idx].coordinates  # [N, 2]
    
    # Normalize features across all samples - GPU accelerated
    feats_flat = feats.reshape(-1, C * H * W)  # [B*T, N]
    feats_mean = feats_flat.mean(dim=0, keepdim=True)
    feats_std = feats_flat.std(dim=0, keepdim=True) + 1e-9
    x = (feats_flat - feats_mean) / feats_std
    
    # Compute autocorrelation - GPU accelerated
    pos_np = pos.cpu().numpy()
    N = pos_np.shape[0]
    
    # Select or randomly choose probe unit
    if probe_idx is None:
        probe_idx = np.random.choice(N)
    
    print(f"Probe unit index: {probe_idx}")
    print(f"Probe unit position: ({pos_np[probe_idx, 0]:.2f}, {pos_np[probe_idx, 1]:.2f})")
    
    # Compute autocorrelation with respect to probe unit
    probe_x = x[:, probe_idx:probe_idx+1]  # [B*T, 1]
    autocorr = torch.mm(probe_x.T, x).flatten() / x.shape[0]  # [N] - GPU
    autocorr[probe_idx] = 0.0  # Set self-correlation to 0
    autocorr_np = autocorr.cpu().numpy()
    
    # Compute distances from probe unit
    distances = np.linalg.norm(pos_np - pos_np[probe_idx], axis=1)
    
    # Unit 1: Probe unit itself
    idx1 = probe_idx
    
    # Unit 2: Strongly correlated unit with some distance
    # Find units with correlation > 0.6 and distance > 1st percentile
    corr_threshold = 0.55
    min_dist = np.percentile(distances[distances > 0], 7)  # Exclude distance 0 (self)
    
    correlated_mask = (autocorr_np > corr_threshold) & (distances < min_dist)
    if np.any(correlated_mask):
        correlated_indices = np.where(correlated_mask)[0]
        # Among correlated units, pick one with moderate distance (not too close, not too far)
        correlated_distances = distances[correlated_indices]
        # Target distance around 30-50th percentile of all distances
        target_dist = np.percentile(distances, 40)
        closest_to_target = np.argmin(np.abs(correlated_distances - target_dist))
        idx2 = correlated_indices[closest_to_target]
    else:
        # Fallback: pick the most correlated unit with any distance
        print(f"Warning: No correlated units found with distance > {min_dist:.2f}")
        valid_indices = np.arange(N)[np.arange(N) != probe_idx]
        idx2 = valid_indices[np.argmax(autocorr_np[valid_indices])]
    
    # Unit 3: Strongly decorrelated unit with more distance
    # Find units with correlation < -0.3 and distance > 50th percentile
    decorr_threshold = -0.3
    min_dist_decorr = np.percentile(distances, 50)
    
    decorrelated_mask = (autocorr_np < decorr_threshold) & (distances > min_dist_decorr)
    if np.any(decorrelated_mask):
        decorrelated_indices = np.where(decorrelated_mask)[0]
        # Among decorrelated units, pick the most decorrelated one
        idx3 = decorrelated_indices[np.argmin(autocorr_np[decorrelated_indices])]
    else:
        # Fallback: pick the most decorrelated unit with larger distance
        print(f"Warning: No decorrelated units found with distance > {min_dist_decorr:.2f}")
        far_units = distances > np.percentile(distances, 70)
        far_indices = np.where(far_units)[0]
        if len(far_indices) > 0:
            idx3 = far_indices[np.argmin(autocorr_np[far_indices])]
        else:
            idx3 = np.argmin(autocorr_np)
    
    selected_indices = [idx1, idx2, idx3]
    print(f"Selected units: {selected_indices}")
    print(f"  Unit 1 (probe): corr={autocorr_np[idx1]:.3f}, dist={distances[idx1]:.2f}")
    print(f"  Unit 2 (correlated): corr={autocorr_np[idx2]:.3f}, dist={distances[idx2]:.2f}")
    print(f"  Unit 3 (decorrelated): corr={autocorr_np[idx3]:.3f}, dist={distances[idx3]:.2f}")
    selected_positions = pos_np[selected_indices]
    
    # Convert activations to CPU for plotting
    feats_np = feats.cpu().numpy()
    selected_activations = feats_np[:, :, selected_indices]  # [B, T, 3]
    
    # Create visualization
    fig = plt.figure(figsize=(9, 4.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
    
    # Left column: Show spatial location of selected units
    ax_spatial = fig.add_subplot(gs[:, 0])

    absvmax = np.max(np.abs(autocorr_np))
    sc = ax_spatial.scatter(pos_np[:, 0], pos_np[:, 1], c=autocorr_np, cmap='RdBu_r',
                           s=1.5, vmin=-absvmax, vmax=absvmax, alpha=1, edgecolors='none', rasterized=True)
    ax_spatial.set_xlim(pos_np[:, 0].min(), pos_np[:, 0].max())
    ax_spatial.set_ylim(pos_np[:, 1].min(), pos_np[:, 1].max())
    
    # Mark selected units with distinct colors

    # get two ends of colormap RdBu_r
    cmap = plt.get_cmap('RdBu_r')
    color_min = cmap(0.12)  # Blue end
    color_max = cmap(0.88)  # Red end

    unit_colors = [color_max, color_max, color_min]
    unit_labels = [
        f'Unit 1 (probe)\ncorr=1.00, d={distances[idx1]:.1f}',
        f'Unit 2 (correlated)\ncorr={autocorr_np[idx2]:.2f}, d={distances[idx2]:.1f}',
        f'Unit 3 (decorrelated)\ncorr={autocorr_np[idx3]:.2f}, d={distances[idx3]:.1f}'
    ]
    for i, (idx, color, label) in enumerate(zip(selected_indices, unit_colors, unit_labels)):
        ax_spatial.scatter(pos_np[idx, 0], pos_np[idx, 1], c=color, s=200,
                          marker='*', linewidths=3, edgecolors='black', 
                          zorder=10, label=label)
    
    ax_spatial.set_xlabel('X Position', fontsize=12)
    ax_spatial.set_ylabel('Y Position', fontsize=12)
    ax_spatial.set_title('Selected Units\n(Autocorrelation Map)', fontsize=13, fontweight='bold')
    ax_spatial.set_aspect('equal')
    ax_spatial.set_facecolor('#f8f8f8')
    ax_spatial.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    # plt.colorbar(sc, ax=ax_spatial, fraction=0.046, pad=0.04, label='Correlation')
    
    # Right column: Plot activations over time for each unit
    time_axis = np.arange(B * T)
    
    for unit_idx, (idx, color, label) in enumerate(zip(selected_indices, unit_colors, unit_labels)):
        ax = fig.add_subplot(gs[unit_idx, 1])
        
        activations = selected_activations[:, :, unit_idx].flatten()  # [B*T]
        
        ax.plot(time_axis, activations, color=color, linewidth=2, alpha=0.8)
        # ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add vertical dashed lines between stimuli (corrected position)
        for b in range(1, B):
            ax.axvline(x=b * T - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xlim(0, B * T - 1)

        ax.set_ylabel('Activation', fontsize=11)
        print(f'{label}\nPosition: ({pos_np[idx, 0]:.2f}, {pos_np[idx, 1]:.2f})')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        
        if unit_idx == 2:  # Last row
            ax.set_xlabel('Time (across stimuli)', fontsize=11)
        else:
            ax.set_xticklabels([])
    
    fig.suptitle(f'Layer {layer_idx} - Unit Activations Over Time and Stimuli (First {B} Stimuli)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    filename = f'layer_{layer_idx}_unit_activations{suffix}.svg' if suffix else f'layer_{layer_idx}_unit_activations.svg'
    plt.savefig(dir_path / filename, bbox_inches='tight', dpi=400, format='svg')
    plt.close(fig)


if __name__ == '__main__':
    all_features, positions = validate_features(MODEL_CKPT)
    viz_dir = PLOTS_DIR / 'plot_autocorr_test'
    viz_dir.mkdir(parents=True, exist_ok=True)

    seed = 32 # For reproducibility
    probe_indices = visualize_random_autocorr(all_features, positions, viz_dir, seed=seed)
    visualize_unit_activations_over_time(all_features, positions, viz_dir, probe_idx=probe_indices[1], layer_idx=0, seed=seed)