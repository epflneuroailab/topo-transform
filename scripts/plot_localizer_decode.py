from config import PLOTS_DIR

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from .analysis_utils import DEFAULT_METHOD_ORDER
from .analysis_utils import METHOD_COLORS
from .analysis_utils import METHOD_LABELS
from .analysis_utils import collect_group_results
from .analysis_utils import resolve_group_names
from .common import HUMAN_C
from .common import LOCALIZER_P_THRESHOLD
from .common import LOCALIZER_T_THRESHOLD
from .common import all_roi_colors
from .get_localizer_decode import Extractor, localizer_decode as localizer_decode_legacy
from .get_localizer_decode_ceiling import localizer_decode_ceiling
from .get_localizers import get_localizer_human, localizers
from .localizer_registry import get_localizer_result_key
from .localizer_registry import get_roi_t_threshold
from .plot_utils import ensure_dir, savefig
from validate.floc.utils.cluster import find_patches
from validate.neural_decoding import decode, make_decoder, run_features
from data.neural_data.collections.tang2025 import get_compilation
from data.neural_data import get_data_loader
from validate import load_transformed_model
from models import vit_transform
from utils import cached
from validate.rois import nsd


_CLUSTER_K = {
    "face": 3,
    "body": 4,
    "place": 1,
    "mt": 1,
    "v6": 1,
    "psts": 1,
}
ROIS = ["face", "place", "body", "v6", "psts"]
ROI_DISPLAY_NAMES = ["Face", "Place", "Body", "V6", "pSTS"]


def _ensure_leading_dim(values):
    if torch.is_tensor(values):
        if values.ndim == 1:
            return values.unsqueeze(0)
        if values.ndim == 2:
            return values.unsqueeze(0)
        return values
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr[None, ...]
    return arr


def _mask_from_top_k_clusters(patches, k, total_units):
    mask = np.zeros(total_units, dtype=bool)
    if not patches:
        return mask
    patches_sorted = sorted(patches, key=lambda patch: patch.area, reverse=True)
    for patch in patches_sorted[:k]:
        mask[patch.unit_indices] = True
    return mask


def _collect_merged_localizer_values(ckpt_name, fwhm_mm, resolution_mm):
    t_vals_dicts, p_vals_dicts, layer_positions = localizers(
        ckpt_name,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
    )

    p_val_dict = {}
    for p_vals in p_vals_dicts:
        p_val_dict.update(p_vals)

    t_val_dict = {}
    for t_vals in t_vals_dicts:
        t_val_dict.update(t_vals)

    return t_val_dict, p_val_dict, layer_positions


def _get_cluster_masks_model(
    rois,
    ckpt_name,
    p_thres=LOCALIZER_P_THRESHOLD,
    t_thres=LOCALIZER_T_THRESHOLD,
    fwhm_mm=2.0,
    resolution_mm=1.0,
):
    t_val_dict, p_val_dict, layer_positions = _collect_merged_localizer_values(
        ckpt_name,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
    )

    masks = []
    for roi in rois:
        key = get_localizer_result_key(roi)
        p_vals = p_val_dict[key]
        t_vals = t_val_dict[key]

        k = _CLUSTER_K.get(roi, 1)
        t_thres_used = get_roi_t_threshold(roi, t_thres)

        layer_masks = []
        for layer_idx, (p_val, t_val) in enumerate(zip(p_vals, t_vals)):
            p_val = _ensure_leading_dim(p_val)
            t_val = _ensure_leading_dim(t_val)
            patches = find_patches(
                positions=layer_positions[layer_idx],
                selectivities=t_val,
                p_values=p_val,
                t_threshold=t_thres_used,
                p_threshold=p_thres,
            )
            total_units = layer_positions[layer_idx].shape[0]
            layer_masks.append(_mask_from_top_k_clusters(patches, k, total_units))
        masks.append(layer_masks)
    return masks


def _localizer_decode_clustered(
    ckpt_name,
    rois,
    num_splits,
    fwhm_mm,
    resolution_mm,
    p_threshold=LOCALIZER_P_THRESHOLD,
    t_threshold=LOCALIZER_T_THRESHOLD,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, epoch = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    masks_model = _get_cluster_masks_model(
        rois,
        ckpt_name,
        p_thres=p_threshold,
        t_thres=t_threshold,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
    )
    masks_human = get_localizer_human(rois)

    batch_size = 16
    ratios = (.9, .1, 0.)
    mask = nsd.get_region_voxels(['high-ventral', 'high-lateral', 'high-dorsal'])
    print("Considering high-level cortex regions only. Voxel counts:", mask.sum())

    all_scores = []
    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        for split in range(num_splits):
            print(f"=== Split {split+1}/{num_splits} ===")
            seed = 42 + split
            trainset, testset, _ = get_compilation(vit_transform, ratios=ratios, seed=seed, type='clip')

            train_loader = get_data_loader(trainset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            test_loader = get_data_loader(testset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            extractor = Extractor()

            train_model_activities, train_neural_activities = run_features(
                model,
                train_loader,
                get_features=extractor,
                device=device,
            )

            test_model_activities, test_neural_activities = run_features(
                model,
                test_loader,
                get_features=extractor,
                device=device,
            )

            def _flatten_acts(acts):
                if isinstance(acts, list):
                    return acts
                if acts.ndim > 2:
                    return acts.reshape(acts.shape[0], -1)
                if acts.ndim == 1:
                    return acts[:, None]
                return acts

            def _select_model_units(acts, unit_mask):
                if isinstance(acts, (list, tuple)):
                    for cand in acts:
                        cand_flat = _flatten_acts(cand)
                        if cand_flat.shape[1] == unit_mask.shape[0]:
                            return cand_flat[:, unit_mask]
                    acts = acts[0]
                acts = _flatten_acts(acts)
                if acts.shape[1] != unit_mask.shape[0]:
                    return acts[:, :0]
                return acts[:, unit_mask]

            train_model_acts = []
            train_neural_acts = []
            test_model_acts = []
            test_neural_acts = []

            for i, roi, mask_model, mask_human in zip(range(len(rois)), rois, masks_model, masks_human):
                print(f"Processing ROI: {roi}")

                train_model_acts_roi = _select_model_units(train_model_activities, mask_model[0])
                test_model_acts_roi = _select_model_units(test_model_activities, mask_model[0])

                train_neural_acts_roi = train_neural_activities[:, 0, mask_human]
                test_neural_acts_roi = test_neural_activities[:, 0, mask_human]

                train_model_acts.append([train_model_acts_roi])
                train_neural_acts.append(train_neural_acts_roi)
                test_model_acts.append([test_model_acts_roi])
                test_neural_acts.append(test_neural_acts_roi)

            decoder = make_decoder('regress', device)

            scores = np.zeros((len(rois), len(rois)))
            for i, roi_a in enumerate(rois):
                for j, roi_b in enumerate(rois):
                    print(f"Decoding ROI: model {roi_a} to human {roi_b}")

                    train_model_roi = [train_model_acts[i]]
                    train_neural_roi = [train_neural_acts[j]]
                    test_model_roi = [test_model_acts[i]]
                    test_neural_roi = [test_neural_acts[j]]

                    if (
                        train_model_roi[0][0].shape[1] == 0
                        or test_model_roi[0][0].shape[1] == 0
                        or train_neural_roi[0].shape[1] == 0
                        or test_neural_roi[0].shape[1] == 0
                    ):
                        scores[i, j] = 0.0
                        continue

                    decode_scores = decode(
                        train_model_roi,
                        train_neural_roi,
                        test_model_roi,
                        test_neural_roi,
                        decoder,
                    )

                    scores[i, j] = decode_scores.mean().item()

            all_scores.append(scores)

    return np.array(all_scores)


def localizer_decode_clustered(ckpt_name, rois, num_splits=1, fwhm_mm=2.0, resolution_mm=1.0):
    import hashlib
    rois_code = hashlib.md5('_'.join(sorted(rois)).encode()).hexdigest()[:8]
    k_code = hashlib.md5('_'.join(f"{k}:{v}" for k, v in sorted(_CLUSTER_K.items())).encode()).hexdigest()[:8]
    return cached(
        f"localizer_decode_clustered_splits{num_splits}_{ckpt_name}_rois{rois_code}_k{k_code}_fwhm{fwhm_mm}_res{resolution_mm}",
        rerun=False,
    )(_localizer_decode_clustered)(
        ckpt_name,
        rois=rois,
        num_splits=num_splits,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
    )


def plot_localizer_decode_legacy(model_ckpts, store_dir=PLOTS_DIR):
    store_dir = ensure_dir(store_dir) if store_dir is not None else None

    num_splits = 1

    all_scores = []
    for model_ckpt in model_ckpts:
        scores = localizer_decode_legacy(model_ckpt, ROIS, num_splits=num_splits, fwhm_mm=2.0, resolution_mm=1.0)
        all_scores.append(scores)

    ceilings = localizer_decode_ceiling(ROIS, folds=10)

    model_scores = []
    human_ceilings = []
    human_ceiling_stds = []
    for r, roi in enumerate(ROIS):
        mean_score = np.array([scores[:, r, r].mean() for scores in all_scores])
        mean_ceiling = ceilings[ROIS.index(roi)].mean(-1)

        model_scores.append(mean_score)
        human_ceilings.append(mean_ceiling)
        human_ceiling_stds.append(ceilings[ROIS.index(roi)].std(-1))

    model_scores = np.array(model_scores)
    human_ceilings = np.array(human_ceilings)
    human_ceiling_stds = np.array(human_ceiling_stds)

    if store_dir is None:
        return model_scores, human_ceilings
        
    plt.figure(figsize=(2.5, 2.6))
    x = np.arange(len(ROIS))
    width = 0.7
    for r, (roi, model_score, human_ceiling, human_ceiling_std) in enumerate(zip(ROIS, model_scores, human_ceilings, human_ceiling_stds)):
        model_score = model_score.mean()
        human_ceiling = human_ceiling.mean()
        _, roi_color = all_roi_colors[roi]
        plt.bar(x[r], model_score, width, color=roi_color, label='Model' if r == 0 else "", alpha=1)
        plt.plot([x[r] - width/2, x[r] + width/2], 
                 [human_ceiling, human_ceiling], 
                 color=HUMAN_C, linestyle='-', linewidth=2, alpha=1, label='Human Ceiling' if r == 0 else "")

    # scatter individual model scores
    for m, scores in enumerate(all_scores):
        ind_scores = [scores[:, r, r].mean() for r in range(len(ROIS))]
        plt.scatter([x], ind_scores, color='k', alpha=1, s=5)

    sns.despine()

    plt.xticks(x, ROI_DISPLAY_NAMES, rotation=45, ha='right')
    plt.ylim(0, 0.75)
    plt.xlim(-.8, len(ROIS)-0.2)
    plt.ylabel('Prediction score (R)')
    plt.tight_layout()
    savefig(store_dir / "localizer_decoding_scores_comparison.svg")

    return model_scores, human_ceilings


def plot_localizer_decode(model_ckpts, store_dir=PLOTS_DIR):
    store_dir = ensure_dir(store_dir) if store_dir is not None else None

    num_splits = 1

    all_scores = []
    for model_ckpt in model_ckpts:
        scores = localizer_decode_clustered(
            model_ckpt,
            ROIS,
            num_splits=num_splits,
            fwhm_mm=2.0,
            resolution_mm=1.0,
        )
        all_scores.append(scores)

    ceilings = localizer_decode_ceiling(ROIS, folds=10)

    model_scores = []
    human_ceilings = []
    human_ceiling_stds = []
    for r, roi in enumerate(ROIS):
        mean_score = np.array([scores[:, r, r].mean() for scores in all_scores])
        mean_ceiling = ceilings[ROIS.index(roi)].mean(-1)

        model_scores.append(mean_score)
        human_ceilings.append(mean_ceiling)
        human_ceiling_stds.append(ceilings[ROIS.index(roi)].std(-1))

    model_scores = np.array(model_scores)
    human_ceilings = np.array(human_ceilings)
    human_ceiling_stds = np.array(human_ceiling_stds)

    if store_dir is None:
        return model_scores, human_ceilings

    plt.figure(figsize=(2.9, 2.6))
    x = np.arange(len(ROIS))
    width = 0.7
    for r, (roi, model_score, human_ceiling, human_ceiling_std) in enumerate(zip(ROIS, model_scores, human_ceilings, human_ceiling_stds)):
        model_score = model_score.mean()
        human_ceiling = human_ceiling.mean()
        _, roi_color = all_roi_colors[roi]
        plt.bar(x[r], model_score, width, color=roi_color, label='Model' if r == 0 else "", alpha=1)
        plt.plot([x[r] - width/2, x[r] + width/2],
                 [human_ceiling, human_ceiling],
                 color=HUMAN_C, linestyle='-', linewidth=2, alpha=1, label='Human Ceiling' if r == 0 else "")

    for m, scores in enumerate(all_scores):
        ind_scores = [scores[:, r, r].mean() for r in range(len(ROIS))]
        plt.scatter([x], ind_scores, color='k', alpha=1, s=5)

    sns.despine()

    plt.xticks(x, ROI_DISPLAY_NAMES, rotation=45, ha='right')
    plt.ylim(0, 0.75)
    plt.xlim(-.8, len(ROIS)-0.2)
    plt.ylabel('Prediction score (R)')
    plt.tight_layout()
    savefig(store_dir / "localizer_decoding_scores_comparison.svg")

    return model_scores, human_ceilings


def main():
    METHOD_ORDER = DEFAULT_METHOD_ORDER
    method_order = resolve_group_names(METHOD_ORDER)
    results = collect_group_results(
        method_order,
        plot_localizer_decode,
        first_kwargs={"store_dir": PLOTS_DIR},
        rest_kwargs={"store_dir": None},
    )

    first_key = method_order[0]
    _, human_scores = results[first_key]
    human_scores = human_scores.mean(axis=0)

    all_scores = []
    for name in method_order:
        scores, _ = results[name]
        all_scores.append(scores.mean(axis=0))

    plt.figure(figsize=(2.2, 2.0))

    width = 0.71
    labels = [METHOD_LABELS[name] for name in method_order]
    colors = [METHOD_COLORS[name] for name in method_order]
    means = [np.mean(scores) for scores in all_scores]

    bars = plt.barh(labels, means, color=colors, height=width, alpha=1)

    # Add white text labels inside bars
    for bar, label in zip(bars, labels):
        plt.text(0.03, bar.get_y() + bar.get_height() / 2, 
                label, ha='left', va='center', color='white', fontsize=10)

    for i, (scores, color) in enumerate(zip(all_scores, colors)):
        x = np.ones(len(scores)) * i
        plt.scatter(scores, x, color='k', alpha=1, s=5)

    # plt.axvline(human_scores.mean(), color=HUMAN_C, linestyle='-', linewidth=2, label='Human Ceiling')
    ci = 1.96 * human_scores.std()
    plt.fill_betweenx([-1, len(labels)], human_scores.mean() - ci, human_scores.mean() + ci, color=HUMAN_C, alpha=0.3, edgecolor='none')

    # test each model against human ceiling
    for i, scores in enumerate(all_scores):
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(scores, human_scores)
        print(f"{labels[i]} vs Human Ceiling: t={t_stat:.3f}, p={p_value:.5f}")
        print(f"Mean {labels[i]} score: {np.mean(scores):.3f}, Human ceiling: {human_scores.mean():.3f}")

    # test ours' improvement percentage and significance against others
    our_scores = all_scores[0]
    for i, scores in enumerate(all_scores[1:]):
        improvement = (our_scores - scores) / np.abs(scores) * 100
        t_stat, p_value = stats.ttest_rel(our_scores, scores)
        print(f"Ours vs {labels[i+1]}: improvement={improvement.mean():.2f}%, t={t_stat:.3f}, p={p_value:.5f}")

    plt.xlabel('Mean prediction score (R)')

    # add ceiling as horizontal zone
    plt.yticks([])  # Remove y-axis labels since they're now inside the bars
    plt.ylim(-.8, len(labels)-0.2)
    plt.xlim(0, 0.6)
    sns.despine()
    savefig(PLOTS_DIR / "localizer_decoding_score_comparison.svg", dpi=300, bbox_inches='tight')

    print(f"Saved localizer decoding score comparison plot to {PLOTS_DIR / 'localizer_decoding_score_comparison.svg'}")


if __name__ == '__main__':
    main()
