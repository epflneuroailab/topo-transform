import numpy as np

from brainscore_vision import load_dataset

from .get_localizers import localizers, get_localizer_human
from utils import cached


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

        d1_rdm = 1 - np.corrcoef(d1)
        d2_rdm = 1 - np.corrcoef(d2)

        iu = np.triu_indices_from(d1_rdm, k=1)
        d1 = d1_rdm[iu]
        d2 = d2_rdm[iu]

        corr = np.corrcoef(d1, d2)[0, 1]
        if spearman_brown:
            corr = (2 * corr) / (1 + corr)
        corrs.append(corr)

    corrs = np.stack(corrs, axis=0)  # [folds]
    return corrs

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
        
    return joint_ceilings  # List of ROI[split]

def _localizer_rsa_ceiling(rois, folds=10):
    masks_human = get_localizer_human(rois)
    joint_ceilings = get_joint_ceiling(CLIP_DATASETS, masks_human, folds=folds)
    return joint_ceilings  # List of ROI[split]

def localizer_rsa_ceiling(rois, folds=10):
    import hashlib
    rois_code = hashlib.md5('_'.join(sorted(rois)).encode()).hexdigest()[:8]
    return cached(f"localizer_rsa_ceiling_folds{folds}_rois{rois_code}", persistent=True)(_localizer_rsa_ceiling)(rois, folds=folds)
    

if __name__ == "__main__":
    # Example usage
    rois = [
        'face',
        'v6',
        'psts',
        'mt',
    ]
    ceilings = localizer_rsa_ceiling(rois, folds=10)
    for roi, ceiling in zip(rois, ceilings):
        mean_ceiling = ceiling.mean()
        std_ceiling = ceiling.std()
        print(f"ROI: {roi}, Ceiling: {mean_ceiling:.4f} ± {std_ceiling:.4f}")