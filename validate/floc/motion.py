import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import copy

from utils import cached
from data import AFD101, ImageNetVid, Kinetics400
from .utils import t_test


# Produce motion index by reporting t-values contrasting Kinetics400 vs imagenet

def motion_index(model, transform, dataset_motion, dataset_static, 
                 batch_size=32, device='cuda', downsampler=None, frames_per_video=24, video_fps=12):

    print(f"Processing {len(dataset_motion)+len(dataset_static)} videos")
    
    datasets = {
        "motion": [],
        "static": [],
    }

    for category, dataset in zip(
        ["motion", "static"], 
        [dataset_motion, dataset_static]
    ):
        datasets[category] = []
        for idx in range(len(dataset)):
            file_path, label, _ = dataset[idx]
            datasets[category].append((file_path, label))

    t_vals_dict, p_vals_dict = t_test(
        model, 
        transform, 
        datasets, 
        contrasts=[(1, -1)],
        batch_size=batch_size, 
        device=device, 
        downsampler=downsampler,
        frames_per_video=frames_per_video, 
        video_fps=video_fps
    )

    return t_vals_dict, p_vals_dict


def localize_motion(model, transform, frames_per_video=24, video_fps=12,
                    batch_size=32, device='cuda', num_samples=256, seed=42, ret_pvals=False):

    motion_dataset = Kinetics400()
    static_dataset = ImageNetVid()

    # Create subsets
    np.random.seed(seed)
    subsamples = np.random.choice(
        len(motion_dataset.valset), 
        size=min(num_samples, len(motion_dataset.valset)), 
        replace=False
    )
    motion_dataset = Subset(
        motion_dataset.valset, 
        subsamples.tolist()
    )

    subsamples = np.random.choice(
        len(static_dataset.valset), 
        size=min(num_samples, len(static_dataset.valset)), 
        replace=False
    )
    static_dataset = Subset(
        static_dataset.valset, 
        subsamples.tolist()
    )

    # Compute motion index
    t_vals_dict = motion_index(
        model, 
        transform, 
        motion_dataset, 
        static_dataset, 
        batch_size=batch_size, 
        device=device,
        downsampler=None,  # Fixed: added missing parameter
    )

    if ret_pvals:
        return t_vals_dict, p_vals_dict
    return t_vals_dict