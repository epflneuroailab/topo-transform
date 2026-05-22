import torch
import numpy as np
import matplotlib.pyplot as plt

from .get_validate_features import validate_features
from config import PLOTS_DIR

from .common import MODEL_CKPT


def plot_distance_similarity(
    features, 
    positions, 
    save_path='distance_similarity.png', 
    n_bins=11, 
    subsample=10000,  # the cost scales O(N^2), so subsample for efficiency
    max_distance=65,  # the stats over this are basically ~0 mean with decreasing variance
):
    """Plot cortical distance vs response similarity with box plots and scatter points"""
    # Convert to numpy and flatten batch dimension
    positions = positions.numpy()  # (N, 2)

    has_time = features.ndim == 5
    if has_time:
        B, T, C, H, W = features.shape
        features = features.reshape(B * T, C * H * W)
    else:
        B, C, H, W = features.shape
        features = features.reshape(B, C * H * W)

    features = features.T  # (N, D)

    # Subsample for efficiency
    if subsample is not None and features.shape[0] > subsample:
        indices = np.random.choice(features.shape[0], subsample, replace=False)
        features = features[indices]
        positions = positions[indices]

    # Normalize features
    features = (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-10)
    
    # Compute pairwise distances and similarities
    cortical_dist = ((positions[:, None, :] - positions[None, :, :])**2).sum(axis=-1)**0.5
    response_sim = (features @ features.T) / features.shape[1]
    
    # Get upper triangle indices (exclude diagonal)
    triu_indices = np.triu_indices_from(cortical_dist, k=1)
    cortical_dist_flat = cortical_dist[triu_indices]
    response_sim_flat = response_sim[triu_indices]
    
    # Filter by max distance if specified
    if max_distance is not None:
        mask = cortical_dist_flat <= max_distance
        cortical_dist_flat = cortical_dist_flat[mask]
        response_sim_flat = response_sim_flat[mask]
    
    # Bin distances
    bins = np.linspace(cortical_dist_flat.min(), cortical_dist_flat.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    bin_indices = np.digitize(cortical_dist_flat, bins)
    
    # Group similarities by bin
    binned_data = [response_sim_flat[bin_indices == i] for i in range(1, len(bins))]
    
    # Plot with improved aesthetics
    fig, ax = plt.subplots(figsize=(4.7, 4))
    
    # Set style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Add scatter points first (so they're behind boxes)
    np.random.seed(42)
    for i, data in enumerate(binned_data):
        data_ = np.array(data)
        if i == 0:
            # report stats
            mean_sim = np.mean(data_)
            std_sim = np.std(data_)
            n_points = len(data_)
            print(f"Distance bin {bin_centers[i]:.1f} mm: mean={mean_sim:.4f}, std={std_sim:.4f}, n={n_points}")

        # test every bin to see if they are significantly different from zero
        from scipy.stats import ttest_1samp
        t_stat, p_value = ttest_1samp(data_, 0)
        print(f"Distance bin {bin_centers[i]:.1f} mm: t-stat={t_stat:.4f}, p-value={p_value:.4f}")

        if len(data) > 0:
            # Subsample points if too many
            max_points = 150
            if len(data) > max_points:
                sample_idx = np.random.choice(len(data), max_points, replace=False)
                data_to_plot = data[sample_idx]
            else:
                data_to_plot = data
            
            # Add jitter to x positions
            x_jitter = np.random.normal(0, bin_width*0.12, len(data_to_plot))
            x_pos = bin_centers[i] + x_jitter
            
            ax.scatter(x_pos, data_to_plot, alpha=0.5, s=5, c='#FA0022', 
                      edgecolors='none', rasterized=True)
    
    # Create box plot
    bp = ax.boxplot(binned_data, positions=bin_centers, widths=bin_width*0.4,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor='#E4E4E4', edgecolor='#1A1A1A', 
                                  linewidth=1.2, alpha=0.9),
                     whiskerprops=dict(color='#1A1A1A', linewidth=1.2),
                     capprops=dict(color='#1A1A1A', linewidth=1.2),
                     medianprops=dict(color='#E63946', linewidth=1.2))
    
    # # Add subtle grid
    # ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
    # ax.set_axisbelow(True)
    
    # Labels and title
    # ax.set_ylim([-1.05, 1.05])
    
    # Improve tick appearance
    ax.tick_params(axis='both', which='major', labelsize=11, 
                   width=1, length=6, color='#1A1A1A')
    ax.tick_params(axis='y', labelsize=11)

    # x ticks
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{b:.1f}" for b in bin_centers], rotation=45, ha='right')
    ax.set_xlabel('Cortical distance (mm)', fontsize=12, labelpad=8)
    ax.set_ylabel('Response correlation', fontsize=12, labelpad=8)
    ax.set_ylim([-0.15, 0.45])

    # Axes width
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='none')
    print(f"Saved to {save_path}")
    plt.close()


def main():
    all_features, positions = validate_features(MODEL_CKPT)
    layer_features = all_features[0]
    positions = positions[0]
    plot_distance_similarity(
        layer_features,
        positions.coordinates,
        PLOTS_DIR / "plot_wiring_cost.svg",
    )


if __name__ == '__main__':
    main()
