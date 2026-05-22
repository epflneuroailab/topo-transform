import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_random_autocorr(layer_features, layer_positions, dir_path, num_probes=5, suffix=''):
    # layer_features: L [B, (T), C, H, W]

    os.makedirs(dir_path, exist_ok=True)

    L = len(layer_features)
    
    for l in range(L):
        lf = layer_features[l]
        
        # Handle variable dimensions [B, C, H, W] or [B, T, C, H, W]
        if lf.ndim == 5:  # [B, T, C, H, W]
            B, T, C, H, W = lf.shape
            lf = lf.reshape(B * T, C, H, W)
        else:  # [B, C, H, W]
            B, C, H, W = lf.shape
        
        # Flatten spatial dimensions
        feats = lf.reshape(-1, C * H * W).cpu().numpy()  # [B*T, N]
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
        
        # Visualize
        fig, axes = plt.subplots(1, num_probes, figsize=(5*num_probes, 4))
        if num_probes == 1:
            axes = [axes]
        
        for i, probe_idx in enumerate(probe_indices):
            probe_pos = pos[probe_idx]  # [2]
            autocorr = autocorrs[i]  # [N]
            ax = axes[i]
            
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=autocorr, cmap='bwr', s=0.1, vmin=-1, vmax=1)
            ax.scatter(probe_pos[0], probe_pos[1], c='black', s=50, marker='x', linewidths=3)
            
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Probe {probe_idx}\n(x={probe_pos[0]:.2f}, y={probe_pos[1]:.2f})')
            ax.set_aspect('equal')

        plt.suptitle(f'Layer {l} - Spatial Autocorrelations', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Add suffix to filename if provided
        filename = f'layer_{l}_autocorr{suffix}.png' if suffix else f'layer_{l}_autocorr.png'
        plt.savefig(dir_path / filename, bbox_inches='tight', dpi=200)
        plt.close(fig)

def validate_autocorr(batch_layer_features, layer_positions, viz_dir, epoch):
    layer_features = [torch.cat(feats, dim=0).cpu() for feats in zip(*batch_layer_features)]
    visualize_random_autocorr(layer_features, layer_positions, viz_dir, suffix=f"_{epoch+1}")