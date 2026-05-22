from config import PLOTS_DIR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .analysis_utils import (
    CKPT_GROUPS,
    DEFAULT_METHOD_ORDER,
    METHOD_COLORS,
    METHOD_LABELS,
    collect_localizer_tvals,
    resolve_group_names,
)
from .common import HUMAN_C
from .common import LOCALIZER_P_THRESHOLD
from .common import LOCALIZER_T_THRESHOLD
from .common import MODEL_C
from .common import MODEL_CKPT
from .get_localizers import get_localizer_human
from .get_localizers import get_localizer_model
from validate.floc.robert import load_robert_tvals
from .get_smoothness import smoothness
from .plot_utils import ensure_dir, savefig


def plot_all_rois(all_t_vals, ckpts, rois, store_dir=None, p_threshold=LOCALIZER_P_THRESHOLD, t_threshold=LOCALIZER_T_THRESHOLD):
    if store_dir is not None:
        store_dir = ensure_dir(store_dir)

    masks_models = [[] for _ in rois]
    for ckpt in ckpts:
        masks_model = get_localizer_model(rois, ckpt, p_thres=p_threshold, t_thres=t_threshold)
        for r, _ in enumerate(rois):
            masks_models[r].append(masks_model[r])
            
    masks_human = get_localizer_human(rois)
    t_vals_robert = load_robert_tvals()

    all_means_model = []
    all_means_human = []
    means_model = []
    means_human = []
    stds_model = []
    stds_human = []
    for i, roi, mask_model, mask_human in zip(range(len(rois)), rois, masks_models, masks_human):
        
        t_vals_models = []
        for t_vals, mask in zip(all_t_vals, mask_model):
            t_vals_model = [t_val[m] for t_val, m in zip(t_vals, mask)]  # layers
            # here just choose the first layer
            t_vals_model = t_vals_model[0]  # shape: (n_units,)
            t_vals_models.append(t_vals_model)

        mean_models = np.array([(t_vals>0).mean() for t_vals in t_vals_models])  # shape: (n_checkpoints)
        mean_humans = np.array([(t_vals[mask_human]>0).mean() for t_vals in t_vals_robert]) # shape: (n_individuals)
        
        # sometimes model have no units in the ROI
        mean_models = np.nan_to_num(mean_models, nan=0.5)

        mean_model = mean_models.mean(0)
        std_model = mean_models.std(0)
        mean_human = mean_humans.mean(0)
        std_human = mean_humans.std(0)

        all_means_model.append(mean_models)
        means_model.append(mean_model)
        stds_model.append(std_model)
        all_means_human.append(mean_humans)
        means_human.append(mean_human)
        stds_human.append(std_human)

    # compute mae between model and human
    mae = np.mean(np.abs(np.array(all_means_model) - np.array(all_means_human).mean(1)[:, None]), axis=0)
    print(f"Mean Absolute Error (mae) between model and human: {mae.mean():.4f}")

    if store_dir is None:
        return mae

    plt.figure(figsize=(3.7, 2.7))
    x = np.arange(len(rois))
    width = 0.4

    plt.bar(x - width/2, means_model, width, yerr=np.array(stds_model), label='Model', capsize=5, color=MODEL_C)
    plt.bar(x + width/2, means_human, width, yerr=np.array(stds_human), label='Human', capsize=5, color=HUMAN_C)

    # report correlation
    from scipy.stats import pearsonr
    corr, pval = pearsonr(means_model, means_human)
    print(f"Correlation between model and human proportions of motion-selective units across ROIs: r={corr:.4f}, p={pval:.4e}")

    # report means across rois for model and human
    print(f"Overall mean proportion of motion-selective units - Model: {np.array(means_model).mean():.4f}, Human: {np.array(means_human).mean():.4f}")

    for i in range(len(rois)):
        plt.scatter([x[i] - width/2]*len(all_means_model[i]), all_means_model[i], color='k', s=10)
        plt.scatter([x[i] + width/2]*len(all_means_human[i]), all_means_human[i], color='k', s=10)

    roi_display_names = [
        'Face',
        'Body',
        'Place',
        'MT',
        'V6',
        'pSTS',
        'V6-enhanced',
        'pSTS-enhanced',
    ]
    if len(rois) == 6:
        plt.xticks(x, roi_display_names[:6])
    else:
        plt.xticks(x, roi_display_names[:len(rois)], rotation=45, ha='right')
    plt.ylabel('Proportion of motion-selective units')
    plt.ylim(0)

    # despine
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    path = store_dir / "localizer_tvals_comparison.svg"
    savefig(path, bbox_inches='tight')
    print(f"Saved localizer t-values comparison plot to {path}")

    return mae


def main():
    store_dir = ensure_dir(PLOTS_DIR / "localizer_motion")

    rois = [
        'face',
        'body',
        'place',
        'mt',
        'v6',
        'psts',
    ]

    METHOD_ORDER = DEFAULT_METHOD_ORDER
    method_order = resolve_group_names(METHOD_ORDER)
    primary_group = "TopoTransform" if "TopoTransform" in method_order else method_order[0]
    primary_ckpts = CKPT_GROUPS[primary_group]

    all_model_results = []
    for i, model_path in enumerate(primary_ckpts):
        ret = smoothness(model_path, 'robert')
        all_model_results.append(ret)
    model_smoothness = []
    human_smoothness = []
    for ret in all_model_results:
        model_s = ret['robert']['model_smoothness']
        human_s = ret['robert']['human_smoothness']
        model_smoothness.append(model_s)
        human_smoothness.append(human_s)
    print(f"Model smoothness: mean={np.mean(model_smoothness):.4f}")
    print(f"Human smoothness: mean={np.mean(human_smoothness):.4f}")
    
    def _maybe_plot_robert_tvals(ckpt_name, t_vals_dicts, p_vals_dicts, layer_positions):
        if ckpt_name != MODEL_CKPT:
            return
        t_vals = t_vals_dicts['robert']
        pos = layer_positions[0]
        v_max_abs = np.max(np.abs(t_vals))
        plt.scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            c=-np.array(t_vals),
            cmap='Spectral',
            s=1,
            norm=Normalize(vmin=-v_max_abs, vmax=v_max_abs),
        )
        plt.colorbar(label='t-value')
        plt.title('Localizer t-values (Robert Dataset)')
        plt.xlabel('Layer Position X')
        plt.ylabel('Layer Position Y')
        plt.gca().set_aspect('equal', adjustable='box')
        savefig(store_dir / "localizer_tvals_robert.png", dpi=400)
        print("Model t vals saved.")

    mae_by_method = {}
    for i, name in enumerate(method_order):
        ckpts = CKPT_GROUPS[name]
        group_store_dir = ensure_dir(store_dir / name.lower())
        all_t_vals = collect_localizer_tvals(
            ckpts,
            dataset='robert',
            ret_merged=True,
            on_result=_maybe_plot_robert_tvals if name == primary_group else None,
        )
        mae_by_method[name] = plot_all_rois(
            all_t_vals,
            ckpts,
            rois,
            group_store_dir,
        )

    plt.figure(figsize=(3.3, 2.7))

    maes = [mae_by_method[name] for name in method_order]
    methods = [METHOD_LABELS[name] for name in method_order]
    values = [mae.mean() for mae in maes]
    colors = [METHOD_COLORS[name] for name in method_order]
    model_mae = maes[0]

    model_results = maes
    y_pos = np.arange(len(methods))
    bars = plt.barh(y_pos, values, color=colors, alpha=1)

    for i, results in enumerate(model_results):
        plt.plot(results, [i] * len(results), 'ko', markersize=4)

    for bar, label in zip(bars, methods):
        plt.text(0.03, bar.get_y() + bar.get_height() / 2, 
                label, ha='left', va='center', color='white', fontsize=10)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.yticks([])
    plt.gca().tick_params(axis='y', length=0)

    plt.xlabel('Mean absolute error', fontsize=12)
    plt.tight_layout()
    savefig(store_dir / "localizer_motion_mae_comparison.svg")

    from scipy.stats import ttest_ind
    for mae, method in zip(maes[1:], methods[1:]):
        t_stat, p_val = ttest_ind(mae, model_mae)
        print(f"MAE comparison between Ours and {method}: Ours mean={model_mae.mean():.4f}, {method} mean={mae.mean():.4f}, t-statistic={t_stat:.4f}, p-value={p_val:.4e}")

    print(f"Saved localizer motion mae comparison plot to {store_dir / 'localizer_motion_mae_comparison.svg'}")

if __name__ == "__main__":
    main()
