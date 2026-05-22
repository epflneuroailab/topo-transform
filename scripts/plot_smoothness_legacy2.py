import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import PLOTS_DIR
from .analysis_utils import (
    METHOD_COLORS,
    METHOD_LABELS,
    collect_group_results,
    resolve_group_names,
    DEFAULT_METHOD_ORDER,
)
from .get_smoothness import smoothness
from .common import HUMAN_C
from .common import MODEL_CKPTS
from .common import UNOPTIMIZED_CKPTS
from .plot_utils import savefig

def plot_smoothness_comparison(model_paths, category='pitcher', fwhm_mm=2.0, resolution_mm=1.0, save_dir=PLOTS_DIR):
    """
    Generate bar plots comparing model and human smoothness across categories,
    with separate bars for moving/static conditions and human reference with CI.
    
    Parameters:
    -----------
    model_paths : str or list of str
        Path(s) to the model checkpoint(s). If multiple paths provided, will compute
        mean and confidence intervals across models.
    category : str
        Category name for analysis
    fwhm_mm : float
        FWHM for smoothing in mm
    resolution_mm : float
        Resolution in mm
    save_dir : str, optional
        Directory to save plots. If None, uses PLOTS_DIR from config
    """
    
    # Convert single model path to list
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    
    # Get smoothness data for all models
    print(f"Computing smoothness for {len(model_paths)} model(s), category: {category}")
    
    all_model_results = []
    for i, model_path in enumerate(model_paths):
        print(f"  Model {i+1}/{len(model_paths)}: {model_path}")
        ret = smoothness(model_path, category, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
        all_model_results.append(ret)
    
    # Prepare data for plotting - aggregate across models
    categories = list(all_model_results[0].keys())
    
    # Collect model smoothness values across all models
    model_smoothness_all = []  # List of lists: [model][category]
    human_smoothness = []  # Human data (same across models)
    
    for cat in categories:
        model_vals = [result[cat]['model_smoothness'] for result in all_model_results]
        model_smoothness_all.append(model_vals)
        # Human smoothness is the same for all models, just take from first
        human_smoothness.append(all_model_results[0][cat]['human_smoothness'])
    
    # Compute statistics across models
    model_smoothness_mean = [np.mean(vals) for vals in model_smoothness_all]
    model_smoothness_std = [np.std(vals) for vals in model_smoothness_all]
    model_smoothness_sem = [stats.sem(vals) if len(vals) > 1 else 0 for vals in model_smoothness_all]
    
    # 95% confidence interval
    confidence_level = 0.95
    model_smoothness_ci = []
    for vals in model_smoothness_all:
        if len(vals) > 1:
            ci = stats.t.interval(confidence_level, len(vals)-1, 
                                 loc=np.mean(vals), 
                                 scale=stats.sem(vals))
            model_smoothness_ci.append((ci[1] - np.mean(vals)))  # Half-width of CI
        else:
            model_smoothness_ci.append(0)
    
    # Compute human confidence interval
    human_mean = np.mean(human_smoothness)
    human_sem = stats.sem(human_smoothness) if len(human_smoothness) > 1 else 0
    if len(human_smoothness) > 1:
        human_ci = stats.t.interval(confidence_level, len(human_smoothness)-1,
                                   loc=human_mean, scale=human_sem)
    else:
        human_ci = (human_mean, human_mean)

    # Print statistics
    overall_model_scores = np.array(model_smoothness_mean)
    overall_human_scores = np.array(human_smoothness)

    # Return data including moving/static breakdown
    return_data = {
        'overall_model': overall_model_scores,
        'overall_human': overall_human_scores,
        'model_by_category': model_smoothness_all,
        'human_by_category': human_smoothness,
        'categories': categories
    }
    
    return return_data


def _build_meta_category_data(results, method_order):
    model_data = results[method_order[0]]

    all_data = {"HUMAN": model_data["overall_human"]}
    for name in method_order:
        all_data[name] = results[name]["overall_model"]

    all_data_meta = {}
    for name in method_order:
        data_dict = results[name]
        moving_vals_per_model = []
        static_vals_per_model = []

        n_models = len(data_dict["model_by_category"][0])

        for model_idx in range(n_models):
            moving_avg = []
            static_avg = []
            for cat_idx, cat in enumerate(data_dict["categories"]):
                val = data_dict["model_by_category"][cat_idx][model_idx]
                if "moving" in cat.lower():
                    moving_avg.append(val)
                elif "static" in cat.lower():
                    static_avg.append(val)

            if moving_avg:
                moving_vals_per_model.append(np.mean(moving_avg))
            if static_avg:
                static_vals_per_model.append(np.mean(static_avg))

        all_data_meta[name] = {
            "moving": moving_vals_per_model,
            "static": static_vals_per_model,
        }

    human_moving = []
    human_static = []
    for i, cat in enumerate(model_data["categories"]):
        if "moving" in cat.lower():
            human_moving.append(model_data["human_by_category"][i])
        elif "static" in cat.lower():
            human_static.append(model_data["human_by_category"][i])
    all_data_meta["HUMAN"] = {"moving": human_moving, "static": human_static}

    return model_data, all_data, all_data_meta


def main():
    ret = plot_smoothness_comparison(UNOPTIMIZED_CKPTS, fwhm_mm=0.0, resolution_mm=1.0)
    # ret = plot_smoothness_comparison(MODEL_CKPTS, fwhm_mm=0.0, resolution_mm=1.0)
    breakpoint()

    METHOD_ORDER = DEFAULT_METHOD_ORDER
    method_order = resolve_group_names(METHOD_ORDER)
    results = collect_group_results(
        method_order,
        plot_smoothness_comparison,
        rest_kwargs={"save_dir": None},
    )
    model_data, all_data, all_data_meta = _build_meta_category_data(results, method_order)

    MODELS = list(method_order)

    # for all models, test moving vs static significance
    print("\n" + "="*50)
    print("MOVING vs STATIC SMOOTHNESS COMPARISON")
    print("="*50)
    for name in MODELS:
        moving = all_data_meta[name]['moving']
        static = all_data_meta[name]['static']
        t_stat, p_value = stats.ttest_ind(moving, static)
        print(f"{name:12s} - Moving mean: {np.mean(moving):.4f}, Static mean: {np.mean(static):.4f}, Diff: {np.mean(moving) - np.mean(static):+.4f}, t={t_stat:.3f}, p={p_value:.5f}")      


    # for all models, show the difference against human 
    print("\n" + "="*50)
    print("MODEL vs HUMAN SMOOTHNESS COMPARISON")
    print("="*50)
    for name in MODELS:
        model_vals = all_data[name]
        human_vals = all_data['HUMAN']
        diff = model_vals - human_vals
        print(f"{name:12s} - Diff mean, std: {np.mean(diff):+.4f}, {np.std(diff):.4f}")

    # Create combined comparison plot with moving/static breakdown
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3))
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    display_labels = ['Human'] + [METHOD_LABELS[name] for name in MODELS]
    
    # Reorder data to match display order (Human first, then models)
    label_order = ['HUMAN'] + MODELS
    
    # Colors: green for Human, blue for MODEL (Ours), gray for others
    colors_moving = [HUMAN_C] + [METHOD_COLORS[name] for name in MODELS]
    colors_static = ['#8FC794'] + [
        '#8DB3D6' if name == 'MODEL' else '#9B9B9B' for name in MODELS
    ]
    
    y_pos = np.arange(len(display_labels))
    bar_height = 0.37
    
    # Calculate means for moving and static separately
    # Each "dot" will be the average over all classes per meta category
    moving_means = []
    static_means = []
    
    for label in label_order:
        # Average across all moving categories
        moving_means.append(np.mean(all_data_meta[label]['moving']))
        # Average across all static categories
        static_means.append(np.mean(all_data_meta[label]['static']))
    
    # Plot horizontal bars - moving (darker) and static (lighter), exclude Human
    bars_moving = ax.barh(y_pos[1:] - bar_height/2, moving_means[1:], height=bar_height,
                          color=colors_moving[1:], alpha=1, edgecolor='none', label='Moving')
    bars_static = ax.barh(y_pos[1:] + bar_height/2, static_means[1:], height=bar_height,
                          color=colors_static[1:], alpha=1, edgecolor='none', label='Static')

    # Human reference as vertical dashed lines across the full height
    ax.axvline(moving_means[0], color=colors_moving[0],
               linestyle='--', linewidth=2.2, zorder=5)
    ax.axvline(static_means[0], color=colors_static[0],
               linestyle='--', linewidth=2.2, zorder=5)
    
    # Add data points as black dots (no jitter, one per meta category)
    dot_offset = 0.12
    for i in range(len(display_labels)):
        if i == 0: continue
        for m in range(n_models):
            # Moving dot
            ax.scatter(all_data_meta[label_order[i]]['moving'][m], y_pos[i] - dot_offset,
                      color='black', s=5, alpha=1, zorder=10, marker='o')
            # Static dot
            ax.scatter(all_data_meta[label_order[i]]['static'][m], y_pos[i] + dot_offset,
                      color='black', s=5, alpha=1, zorder=10, marker='o')
    
    # Customize plot
    ax.set_xlabel('Spatial autocorrelation (Moran\'s I)', fontsize=10, fontweight='normal')
    ax.set_yticks([])
    ax.set_xlim(0, None)
    
    # Add white text labels on bars (centered between moving/static)
    for i, label in enumerate(display_labels):
        if i == 0: continue
        ax.text(0.02, y_pos[i], label,
               va='center', ha='left', fontsize=10, color='white')
        ax.text(0.27, y_pos[i]-bar_height/2, 'moving',
               va='center', ha='left', fontsize=8, color='white')
        ax.text(0.27, y_pos[i]+bar_height/2, 'static',
               va='center', ha='left', fontsize=8, color='white')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    savefig(
        PLOTS_DIR / "smoothness_comparison_bar.svg",
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
    )
    
    print(f"\nSaved smoothness comparison plot to {PLOTS_DIR / f'smoothness_comparison_bar.svg'}")


if __name__ == '__main__':
    main()
