import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict
from tqdm import tqdm
import copy
import os

from utils import cached
from data import Kinetics400
from .categories import KANWISHER
from .utils import t_test, CategoryDataset


KUCUK2024 = "/mnt/scratch/ytang/datasets/fsaverage_surfaces_pitcher"

class MovingDataset(CategoryDataset):
    def __init__(self, *args, mode='moving', seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.seed = seed
    
    def __getitem__(self, idx):
        # Get item from base dataset (always in normal ordering)
        data, label = super().__getitem__(idx)

        # data shape: (T, C, H, W)
        num_frames = data.shape[0]
        
        # Apply ordering transformation
        if self.mode == 'static':
            # repeat the first, middle, and the last frame, one for each third of the video
            assert num_frames % 3 == 0, "Number of frames must be divisible by 3 for static mode."
            third = num_frames // 3
            first_frame = data[0:1].repeat(third, 1, 1, 1)
            middle_frame = data[third:third+1].repeat(third, 1, 1, 1)
            last_frame = data[-1:].repeat(num_frames - 2*third, 1, 1, 1)
            data = torch.cat([first_frame, middle_frame, last_frame], dim=0)

        return data, label

def KANWISHER_dynamic_static_category_dataset(data_dir=KANWISHER, transform=None, frames_per_video=36, video_fps=12):
    """Create a category dataset for the Kanwisher dataset."""
    datasets = defaultdict(list)
    for category in os.listdir(data_dir):
        if category == "Scrambled15G": # skip scrambled images
            continue
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for fname in os.listdir(category_dir):
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                    datasets[category].append((os.path.join(category_dir, fname), category))

    ret = {}
    for cate, file_infos in datasets.items():
        ret[cate + '_moving'] = MovingDataset(file_infos, transform=transform, frames_per_video=frames_per_video, video_fps=video_fps, mode='moving')
        ret[cate + '_static'] = MovingDataset(file_infos, transform=transform, frames_per_video=frames_per_video, video_fps=video_fps, mode='static')
    
    return ret

def localize_pitcher(model, transform, frames_per_video=36, video_fps=12,
                     batch_size=32, device='cuda', num_samples=256, seed=42, downsampler=None, ret_pvals=False):
    
    categories = [
        "Faces_moving", "Faces_static",
        "Bodies_moving", "Bodies_static",
        "Scenes_moving", "Scenes_static",
        "Objects_moving", "Objects_static",
    ]

    # Group files by category
    datasets = KANWISHER_dynamic_static_category_dataset(transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)
    datasets = {cat: datasets[cat] for cat in categories}
    n_categories = len(categories)
    t_vals_dict, p_vals_dict = t_test(
        model, transform, 
        datasets=datasets, 
        contrasts=[
            # dynamic vs static
            (1, -1, 0, 0, 0, 0, 0, 0),   # Faces moving vs static
            (0, 0, 1, -1, 0, 0, 0, 0),   # Bodies moving vs static
            (0, 0, 0, 0, 1, -1, 0, 0),   # Scenes moving vs static
            (0, 0, 0, 0, 0, 0, 1, -1),   # Objects moving vs static
            # individual conditions
            (1, 0, 0, 0, 0, 0, 0, 0),    # Faces moving overall
            (0, 1, 0, 0, 0, 0, 0, 0),    # Faces static overall
            (0, 0, 1, 0, 0, 0, 0, 0),    # Bodies moving overall
            (0, 0, 0, 1, 0, 0, 0, 0),    # Bodies static overall
            (0, 0, 0, 0, 1, 0, 0, 0),    # Scenes moving overall
            (0, 0, 0, 0, 0, 1, 0, 0),    # Scenes static overall
            (0, 0, 0, 0, 0, 0, 1, 0),    # Objects moving overall
            (0, 0, 0, 0, 0, 0, 0, 1),    # Objects static overall
            # localizers
            (1, 0, 0, 0, 0, 0, -1, 0),  # dynamic faces vs dynamic objects
            (0, 1, 0, 0, 0, 0, 0, -1),  # static faces vs static objects
            (0, 0, 1, 0, 0, 0, -1, 0),  # dynamic bodies vs dynamic objects
            (0, 0, 0, 1, 0, 0, 0, -1),  # static bodies vs static objects
            (0, 0, 0, 0, 1, 0, -1, 0),  # dynamic scenes vs dynamic objects
            (0, 0, 0, 0, 0, 1, 0, -1),  # static scenes vs static objects
        ],
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )

    # Use contrast indices (0-3) to access results
    t_vals_ret = {
        "Faces_moving_vs_static": t_vals_dict["Faces_moving_vs_Faces_static"],
        "Bodies_moving_vs_static": t_vals_dict["Bodies_moving_vs_Bodies_static"],
        "Scenes_moving_vs_static": t_vals_dict["Scenes_moving_vs_Scenes_static"],
        "Objects_moving_vs_static": t_vals_dict["Objects_moving_vs_Objects_static"],
        "Faces_moving": t_vals_dict["Faces_moving_vs_baseline"],
        "Faces_static": t_vals_dict["Faces_static_vs_baseline"],
        "Bodies_moving": t_vals_dict["Bodies_moving_vs_baseline"],
        "Bodies_static": t_vals_dict["Bodies_static_vs_baseline"],
        "Scenes_moving": t_vals_dict["Scenes_moving_vs_baseline"],
        "Scenes_static": t_vals_dict["Scenes_static_vs_baseline"],
        "Objects_moving": t_vals_dict["Objects_moving_vs_baseline"],
        "Objects_static": t_vals_dict["Objects_static_vs_baseline"],
    }

    for category in categories:
        for k in t_vals_dict.keys():
            if k.startswith(category):
                vs_after = k.split('_vs_')[1]
                vs_before = k.split('_vs_')[0]
                if 'baseline' in vs_after:
                    continue
                if category in vs_after:
                    continue
                t_vals_ret[f"{vs_before}_localizer"] = t_vals_dict[k]

    p_vals_ret = {
        "Faces_moving_vs_static": p_vals_dict["Faces_moving_vs_Faces_static"],
        "Bodies_moving_vs_static": p_vals_dict["Bodies_moving_vs_Bodies_static"],
        "Scenes_moving_vs_static": p_vals_dict["Scenes_moving_vs_Scenes_static"],
        "Objects_moving_vs_static": p_vals_dict["Objects_moving_vs_Objects_static"],
        "Faces_moving": p_vals_dict["Faces_moving_vs_baseline"],
        "Faces_static": p_vals_dict["Faces_static_vs_baseline"],
        "Bodies_moving": p_vals_dict["Bodies_moving_vs_baseline"],
        "Bodies_static": p_vals_dict["Bodies_static_vs_baseline"],
        "Scenes_moving": p_vals_dict["Scenes_moving_vs_baseline"],
        "Scenes_static": p_vals_dict["Scenes_static_vs_baseline"],
        "Objects_moving": p_vals_dict["Objects_moving_vs_baseline"],
        "Objects_static": p_vals_dict["Objects_static_vs_baseline"],
    }

    for category in categories:
        for k in p_vals_dict.keys():
            if k.startswith(category):
                vs_after = k.split('_vs_')[1]
                vs_before = k.split('_vs_')[0]
                if 'baseline' in vs_after:
                    continue
                if category in vs_after:
                    continue
                p_vals_ret[f"{vs_before}_localizer"] = p_vals_dict[k]

    if ret_pvals:
        return t_vals_ret, p_vals_ret
    return t_vals_ret

def localize_pitcher_human():
    categories = ["Faces", "Bodies", "Scenes", "Objects"]
    conditions = {"moving": "dyn", "static": "stat"}

    ret = {}
    for category in categories:
        for cond_name, cond_suffix in conditions.items():
            roi_name = f"{category.lower()}{cond_suffix}:t-stat.npy"
            filepath = os.path.join(KUCUK2024, roi_name)
            t_vals = np.load(filepath, allow_pickle=True)
            ret[f"{category}_{cond_name}"] = t_vals

    return ret