from config import PLOTS_DIR

import os
import numpy as np
import matplotlib.pyplot as plt

from config import ROBERT_STATS
from .analysis_utils import CKPT_GROUPS, resolve_group_names
from .common import HUMAN_C
from .common import MODEL_C
from .get_localizers import localizers


def load_robert_tvals():
    t_vals = []
    for individual in os.listdir(ROBERT_STATS):
        if not individual.endswith('.npy'):
            continue
        t_val = np.load(os.path.join(ROBERT_STATS, individual))
        t_vals.append(t_val)
    t_vals = np.array(t_vals)
    t_vals_mean = t_vals.mean(0)
    return t_vals_mean


def _plot_hist(values, color, title, save_path, label):
    plt.figure(figsize=(5, 2.7) if "model" in title.lower() else (3, 2))
    weights = np.ones_like(values) / len(values)
    plt.hist(values, bins=50, alpha=1, label=label, weights=weights, color=color)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('t-value')
    plt.ylabel('Probability')
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    store_dir = PLOTS_DIR
    store_dir.mkdir(parents=True, exist_ok=True)

    METHOD_ORDER = ("TopoTransform",)
    method_order = resolve_group_names(METHOD_ORDER)
    primary_group = method_order[0]

    t_vals_model = []
    for ckpt_name in CKPT_GROUPS[primary_group]:
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, ret_merged=True)
        t_vals_model.append(t_vals_dicts['robert'][0].flatten())

    t_vals_model = np.array(t_vals_model).mean(axis=0)
    t_vals_human = load_robert_tvals()

    # report mean std of human and model t-vals
    print(f"Human t-vals range: mean={t_vals_human.mean():.4f}, std={t_vals_human.std():.4f}")
    print(f"Model t-vals range: mean={t_vals_model.mean():.4f}, std={t_vals_model.std():.4f}")

    from validate.rois.nsd import get_region_voxels

    high_level = get_region_voxels([
        'high-dorsal', 
        'high-ventral', 
        'high-lateral',
    ])
    t_vals_human = t_vals_human[high_level].flatten()

    # t_vals_human = t_vals_human/(t_vals_human**2).mean()**0.5
    # t_vals_model = t_vals_model/(t_vals_model**2).mean()**0.5

    import diptest
    model_stat, model_p = diptest.diptest(t_vals_model.flatten())
    print(f"Model t-vals dip test: statistic={model_stat:.4f}, p-value={model_p:.4f}")
    _plot_hist(
        t_vals_model,
        MODEL_C,
        'Distribution of model t-values',
        store_dir / "robert_tval_distribution.svg",
        "Model",
    )
    print(f"Saved robert t-value distribution plot to {store_dir / 'robert_tval_distribution.svg'}")

    human_stat, human_p = diptest.diptest(t_vals_human.flatten())
    print(f"Human t-vals dip test: statistic={human_stat:.4f}, p-value={human_p:.4f}")
    _plot_hist(
        t_vals_human,
        HUMAN_C,
        'Distribution of human t-values',
        store_dir / "robert_tval_distribution_human.svg",
        "Human",
    )
    print(f"Saved robert t-value distribution plot to {store_dir / 'robert_tval_distribution_human.svg'}")


if __name__ == "__main__":
    main()
