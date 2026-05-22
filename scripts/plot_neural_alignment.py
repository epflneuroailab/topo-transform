from config import PLOTS_DIR

import numpy as np
import matplotlib.pyplot as plt
from .get_neural_alignment import neural_alignment
from .get_task_performance import task_performance

from .analysis_utils import CKPT_GROUPS, resolve_group_names
from .common import DEFAULT_C
from .common import MODEL_C


def plot_combined_comparison(ckpt_names, num_splits=1, figsize=(3.4, 3.4)):
    """
    Plot combined comparison of neural alignment and task performance across multiple checkpoints.
    
    Args:
        ckpt_names: List of checkpoint names or single checkpoint name
        num_splits: Number of cross-validation splits for neural alignment
        figsize: Figure size (width, height)
    """
    # Ensure ckpt_names is a list
    if isinstance(ckpt_names, str):
        ckpt_names = [ckpt_names]
    
    num_models = len(ckpt_names)
    
    # Collect data for all models
    neural_pre_all = []
    neural_post_all = []
    imagenet_pre_all = []
    imagenet_post_all = []
    ssv2_pre_all = []
    ssv2_post_all = []
    
    for ckpt_name in ckpt_names:
        # Neural alignment
        scores_pre, scores_post, mask, ceiling = neural_alignment(ckpt_name, num_splits=num_splits)
        ceiling_val = ceiling[:, mask].mean()
        neural_pre_all.append(scores_pre.mean() / ceiling_val)
        neural_post_all.append(scores_post.mean() / ceiling_val)
        
        # Task performance
        scores_pre, scores_post = task_performance(ckpt_name, "imagenet")
        imagenet_pre_all.append(scores_pre)
        imagenet_post_all.append(scores_post)
        
        scores_pre, scores_post = task_performance(ckpt_name, "ssv2")
        ssv2_pre_all.append(scores_pre)
        ssv2_post_all.append(scores_post)
    
    # Convert to arrays
    neural_pre_all = np.array(neural_pre_all)
    neural_post_all = np.array(neural_post_all)
    imagenet_pre_all = np.array(imagenet_pre_all)
    imagenet_post_all = np.array(imagenet_post_all)
    ssv2_pre_all = np.array(ssv2_pre_all)
    ssv2_post_all = np.array(ssv2_post_all)
    
    # Prepare data
    metrics = ['neural\nalignment', 'object\nrecognition', 'action\nrecognition']
    pre_data = [neural_pre_all, imagenet_pre_all, ssv2_pre_all]
    post_data = [neural_post_all, imagenet_post_all, ssv2_post_all]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#959595', '#FA0022']  # Gray for original, red for transformed
    bar_height = 0.6
    y_positions = np.arange(len(metrics)) * 2  # Space out the groups
    
    # Plot bars
    for i, (metric, pre, post) in enumerate(zip(metrics, pre_data, post_data)):
        y_pre = y_positions[i] + bar_height/2
        y_post = y_positions[i] - bar_height/2
        
        # Compute means and stds
        mean_pre = pre.mean()
        mean_post = post.mean()
        std_pre = pre.std()
        std_post = post.std()
        
        # Plot bars
        ax.barh(y_pre, mean_pre, height=bar_height, color=colors[0], alpha=1, edgecolor='none')
        ax.barh(y_post, mean_post, height=bar_height, color=colors[1], alpha=1, edgecolor='none')
        
        # # Error bars
        # if num_models > 1:
        #     ax.errorbar(mean_pre, y_pre, xerr=std_pre, fmt='none', color='black', 
        #                linewidth=2, capsize=4, capthick=2, zorder=4)
        #     ax.errorbar(mean_post, y_post, xerr=std_post, fmt='none', color='black', 
        #                linewidth=2, capsize=4, capthick=2, zorder=4)
        
        # Individual model points
        if num_models > 1:
            y_jitter_pre = np.array([y_pre] * len(pre))
            y_jitter_post = np.array([y_post] * len(post))
            ax.scatter(pre, y_jitter_pre, color='black', s=17, alpha=1, 
                      zorder=5, edgecolors='none')
            ax.scatter(post, y_jitter_post, color='black', s=17, alpha=1, 
                      zorder=5, edgecolors='none')
        
        # T test of pre vs post
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(pre, post)
        print(f"{metric} T-test: t-stat={t_stat:.4f}, p-value={p_value:.4f}")

        # Add metric label
        ax.text(-0.04, y_positions[i], metric, ha='right', va='center', 
                fontsize=10, transform=ax.get_yaxis_transform())
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Original'),
        Patch(facecolor=colors[1], label='Transformed')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=9)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel('Performance', fontsize=12)
    ax.set_ylim(-1, y_positions[-1] + 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.tick_params(left=False)
    
    plt.tight_layout()
    
    return fig, ax


def main():
    METHOD_ORDER = ("TopoTransform",)
    method_order = resolve_group_names(METHOD_ORDER)
    ckpt_names = []
    for name in method_order:
        ckpt_names.extend(CKPT_GROUPS[name])
    
    fig, ax = plot_combined_comparison(ckpt_names, num_splits=1)
    
    plt.savefig(PLOTS_DIR / 'plot_combined_comparison.svg', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {PLOTS_DIR / 'plot_combined_comparison.svg'}")


if __name__ == "__main__":
    main()
