import numpy as np

def correlation_score(x, y):
    x_z = (x - x.mean(0)) / (x.std(0) + 1e-8)
    y_z = (y - y.mean(0)) / (y.std(0) + 1e-8)
    return (x_z * y_z).mean(0)

def compute_ceiling(bs_assembly, folds=10, seed=0, spearman_brown=True):
    
    # reshape assembly to standard dims
    bs_assembly = bs_assembly.reset_index('presentation')
    repetition_vals = np.unique(bs_assembly.coords['repetition'].values)
    num_rep = len(repetition_vals)
    data = []
    for rep in repetition_vals:
        asm = bs_assembly.isel(presentation=bs_assembly.repetition == rep).transpose('presentation', 'time_bin', 'neuroid').values # stupid xarray
        asm = asm.reshape(-1, asm.shape[-1])  # [stimuli x time_bins, units]
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

def compute_joint_ceiling(assemblies, **kwargs):

    ceiling_data = []
    for assembly in assemblies:
        ceiling_data.append((
            assembly.meta['ceiling'],
            assembly.meta['time_bin_duration'], 
            assembly.meta['num_time_bins'],
            assembly.meta['num_presentations'],
        ))

    prod = 0
    n = 0
    for (ceiling, time_bin_duration, num_time_bins, num_presentations) in ceiling_data:
        c = ceiling
        time_pres = time_bin_duration * num_presentations * num_time_bins
        prod += c * time_pres
        n += time_pres

    joint_ceiling = prod / n

    return joint_ceiling  # [split, neuroid:20484]