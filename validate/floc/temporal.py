import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import copy

from utils import cached
from data import Kinetics400
from .utils import t_test, CategoryDataset


class OrderingDataset(CategoryDataset):
    def __init__(self, *args, ordering='normal', seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.ordering = ordering
        self.seed = seed
    
    def __getitem__(self, idx):
        # Get item from base dataset (always in normal ordering)
        data, label = super().__getitem__(idx)

        # data shape: (T, C, H, W)
        num_frames = data.shape[0]
        
        # Apply ordering transformation
        if self.ordering == 'shuffled':
            # Use deterministic shuffling based on seed and idx
            rng = np.random.RandomState(self.seed + idx)
            indices = torch.from_numpy(rng.permutation(num_frames))
            data = data[indices]
        elif self.ordering == 'static':
            # Use only the middle frame, repeated
            middle_idx = num_frames // 2
            data = data[middle_idx:middle_idx+1].repeat(num_frames, 1, 1, 1)
        # else: 'normal' - keep original ordering
        return data, label


def localize_temporal(model, transform, frames_per_video=24, video_fps=12,
                      batch_size=32, device='cuda', num_samples=256, seed=42, ret_pvals=False):
    
    dataset = Kinetics400()

    np.random.seed(seed)
    subsamples = np.random.choice(
        len(dataset.valset), 
        size=min(num_samples, len(dataset.valset)), 
        replace=False
    )
    dataset = Subset(
        dataset.valset, 
        subsamples.tolist()
    )

    print(f"Processing {len(dataset)} videos with seed {seed}")
    print("Extracting features for each ordering...")
    
    # Extract features for each ordering
    orderings = ['normal', 'shuffled', 'static']
    datasets = {}
    
    print("Averaging over time dimension for each video.")
    for ordering in orderings:
        print(f"\nExtracting features for {ordering} ordering...")
        
        # Create wrapped dataset with specific ordering
        wrapped_dataset = OrderingDataset(dataset, transform, ordering=ordering, seed=seed)
        datasets[ordering] = wrapped_dataset
    
    t_vals_dict, p_vals_dict = t_test(
        model, 
        transform, 
        datasets, 
        contrasts=[(1, -1, 0), (1, 0, -1), (0, 1, -1)],
        batch_size=batch_size, 
        device=device, 
        downsampler=None,
        frames_per_video=frames_per_video, 
        video_fps=video_fps,
        related=True,
    )

    t_vals_ret = {
        "motion_vs_static_v2": t_vals_dict["normal_vs_static"],
        "motion_vs_static_v3": t_vals_dict["shuffled_vs_static"],
        "high_lvl_motion": t_vals_dict["normal_vs_shuffled"],
    }
    p_vals_ret = {
        "motion_vs_static_v2": p_vals_dict["normal_vs_static"],
        "motion_vs_static_v3": p_vals_dict["shuffled_vs_static"],
        "high_lvl_motion": p_vals_dict["normal_vs_shuffled"],
    }

    if ret_pvals:
        return t_vals_ret, p_vals_ret

    return t_vals_ret