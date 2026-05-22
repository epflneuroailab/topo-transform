import numpy as np

from brainscore_vision import load_dataset

from .get_localizers import localizers, get_localizer_human
from data.neural_data.utils import correlation_score
from utils import cached

from validate.neural_decoding import decode, make_decoder, run_features


CLIP_DATASETS = [
    "McMahon2023-fMRI",
    "Lahner2024-fMRI",
]

def compute_ceiling(bs_assembly, mask, folds=10, seed=0, spearman_brown=True):
    
    # reshape assembly to standard dims
    bs_assembly = bs_assembly.reset_index('presentation')
    repetition_vals = np.unique(bs_assembly.coords['repetition'].values)
    num_rep = len(repetition_vals)
    data = []
    for rep in repetition_vals:
        asm = bs_assembly.isel(presentation=bs_assembly.repetition == rep).transpose('presentation', 'time_bin', 'neuroid').values # stupid xarray
        # time_bins == 1
        assert asm.shape[1] == 1, "Expected time_bins to be 1"
        asm = asm.reshape(-1, asm.shape[-1])  # [stimuli x time_bins, units]
        asm = asm[:, mask]  # apply mask
        data.append(asm)
    data = np.stack(data, axis=-1)  # [stimuli x time_bins, units, repetitions]

    if num_rep < 2:
        raise ValueError("Need at least 2 repetitions for ceiling computation")

    corrs = []
    for sd in range(seed, seed + folds):
        np.random.seed(sd)
        perm = np.random.permutation(num_rep)
        half1, half2 = perm[: num_rep // 2], perm[num_rep // 2 :]
        d1 = data[..., half1].mean(-1)
        d2 = data[..., half2].mean(-1)
        corr = correlation_score(d1, d2)
        if spearman_brown:
            corr = (2 * corr) / (1 + corr)
        corrs.append(corr)

    corrs = np.stack(corrs, axis=0)  # [folds, units]
    return corrs

def compute_cross_decoding(bs_assembly, masks, seed=42, device='cuda'):
    # reshape assembly to standard dims
    bs_assembly = bs_assembly.reset_index('presentation')
    repetition_vals = np.unique(bs_assembly.coords['repetition'].values)
    num_rep = len(repetition_vals)
    data = []
    for rep in repetition_vals:
        asm = bs_assembly.isel(presentation=bs_assembly.repetition == rep).transpose('presentation', 'time_bin', 'neuroid').values # stupid xarray
        # time_bins == 1
        assert asm.shape[1] == 1, "Expected time_bins to be 1"
        asm = asm.reshape(-1, asm.shape[-1])  # [stimuli x time_bins, units]
        data.append(asm)
    data = np.stack(data, axis=-1)  # [stimuli x time_bins, units, repetitions]

    if num_rep < 2:
        raise ValueError("Need at least 2 repetitions for ceiling computation")

    np.random.seed(seed)
    perm = np.random.permutation(num_rep)
    half1, half2 = perm[: num_rep // 2], perm[num_rep // 2 :]
    d1 = data[..., half1].mean(-1)
    d2 = data[..., half2].mean(-1)

    model_data = []  # as from d1
    human_data = []  # as from d2
    for mask in masks:
        d2_masked = d2[:, mask]
        human_data.append(d2_masked)
        d1_masked = d1[:, mask]
        model_data.append(d1_masked)
    
    decoder = make_decoder('regress', device)
    scores = np.zeros((len(masks), len(masks)))
    for i in range(len(masks)):
        for j in range(len(masks)):
            train_model_roi = [[model_data[i]]]
            train_neural_roi = [human_data[j]]
            test_model_roi = [[model_data[i]]]
            test_neural_roi = [human_data[j]]

            decode_scores = decode(
                train_model_roi,
                train_neural_roi,
                test_model_roi,
                test_neural_roi,
                decoder,
            )

            scores[i, j] = decode_scores.mean().item()  # average over all voxels
    return scores  # [roi_model, roi_human]


def get_ceiling(assembly_name, mask, folds=10):
    assembly = load_dataset(assembly_name)
    ceiling = compute_ceiling(assembly, mask=mask, folds=folds)
    num_samples = assembly.sizes['presentation']
    return ceiling, num_samples

def get_joint_ceiling(assembly_names, masks, folds=10):
    joint_ceilings = []
    for mask in masks:
        ceilings = []
        nums_samples = []
        for name in assembly_names:
            ceiling, num_samples = get_ceiling(name, mask=mask, folds=folds)
            ceilings.append(ceiling)
            nums_samples.append(num_samples)
        prod = 0
        n = 0
        for ceiling, num_samples in zip(ceilings, nums_samples):
            c = ceiling
            time_pres = num_samples  # time_bin_duration == 1, num_time_bins == 1
            prod += c * time_pres
            n += time_pres

        joint_ceiling = prod / n
        joint_ceilings.append(joint_ceiling)
        
    return joint_ceilings  # List of ROI[split, units]

def _localizer_decode_ceiling(rois, folds=10):
    masks_human = get_localizer_human(rois)
    joint_ceilings = get_joint_ceiling(CLIP_DATASETS, masks_human, folds=folds)
    return joint_ceilings  # List of ROI[split, units]

def localizer_decode_ceiling(rois, folds=10):
    import hashlib
    rois_code = hashlib.md5('_'.join(sorted(rois)).encode()).hexdigest()[:8]
    return cached(f"localizer_decode_ceiling_folds{folds}_rois{rois_code}", persistent=True)(_localizer_decode_ceiling)(rois, folds=folds)
    

if __name__ == "__main__":
    # Example usage
    rois = [
        'face',
        'place',
        'body',
        'v6',
        'psts',
    ]
    ceilings = localizer_decode_ceiling(rois, folds=10)
    for roi, ceiling in zip(rois, ceilings):
        mean_ceiling = ceiling.mean(-1).mean()
        std_ceiling = ceiling.mean(-1).std()
        print(f"ROI: {roi}, Ceiling: {mean_ceiling:.4f} ± {std_ceiling:.4f}")

    cross_decoding_scores = compute_cross_decoding(
        load_dataset(CLIP_DATASETS[0]),
        masks=[get_localizer_human([roi])[0] for roi in rois],
        seed=42,
        device='cuda',
    )

    # human self decoding:
    # array([[0.78825743, 0.75453885, 0.79069443, 0.62405491, 0.80807924],
    #        [0.75200228, 0.83395236, 0.76675489, 0.72158678, 0.76914687],
    #        [0.72925461, 0.78765057, 0.86494881, 0.71715551, 0.87376763],
    #        [0.50442148, 0.70740531, 0.60075526, 0.63300629, 0.68025643],
    #        [0.70219325, 0.66971737, 0.7986877 , 0.5082216 , 0.83789202]])