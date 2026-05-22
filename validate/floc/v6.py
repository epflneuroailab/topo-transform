import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import os
import numpy as np
from scipy import stats

from config import FLOW
from utils import cached
from .utils import t_test, CategoryDataset


class MovingDataset(CategoryDataset):
    def __init__(self, *args, mode='moving', **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
    
    def __getitem__(self, idx):
        # Get item from base dataset (always in normal ordering)
        data, label = super().__getitem__(idx)

        # data shape: (T, C, H, W)
        num_frames = data.shape[0]
        
        # Apply ordering transformation
        if self.mode == 'static':
            # repeat the middle frame for the entire video
            data = data[num_frames // 2:num_frames // 2 + 1].repeat(num_frames, 1, 1, 1)

        return data, label


def Pitzalis_category_dataset(data_dir=FLOW, transform=None, frames_per_video=24, video_fps=12):
    """Create a category dataset for the Pitzalis dataset."""
    file_infos = defaultdict(list)
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for fname in os.listdir(category_dir):
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                    file_infos[category].append((os.path.join(category_dir, fname), category))
                    if category == "coherent":
                        file_infos["static"].append((os.path.join(category_dir, fname), "static"))

    datasets = {
        "coherent": MovingDataset(file_infos['coherent'], mode='moving', transform=transform, frames_per_video=frames_per_video, video_fps=video_fps),
        "scrambled": MovingDataset(file_infos['scrambled'], mode='moving', transform=transform, frames_per_video=frames_per_video, video_fps=video_fps),
        "static": MovingDataset(file_infos['static'], mode='static', transform=transform, frames_per_video=frames_per_video, video_fps=video_fps),
    }

    return datasets

def localize_v6(model, transform, 
                batch_size=32, device='cuda', downsampler=None,
                video_fps=12, frames_per_video=24, ret_pvals=False):

    categories = ["coherent", "scrambled", "static"]

    # Group files by category
    datasets = Pitzalis_category_dataset(transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)
    datasets = {cat: datasets[cat] for cat in categories}
    n_categories = len(categories)
    t_vals_dict, p_vals_dict = t_test(
        model, transform, 
        datasets=datasets, contrasts=[(1, -1, 0), (1, -1, -1), (1, 0, -1)],
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )

    t_vals_ret = {
        "V6": t_vals_dict["coherent_vs_scrambled"],
        "V6-enhanced": t_vals_dict["coherent_vs_scrambled+static"],
        "MT-Huk": t_vals_dict["coherent_vs_static"],
    }

    p_vals_ret = {
        "V6": p_vals_dict["coherent_vs_scrambled"],
        "V6-enhanced": p_vals_dict["coherent_vs_scrambled+static"],
        "MT-Huk": p_vals_dict["coherent_vs_static"],
    }

    if ret_pvals:
        return t_vals_ret, p_vals_ret
    return t_vals_ret

    
