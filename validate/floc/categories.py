import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from config import VPNL, KANWISHER
import os
import numpy as np
from scipy import stats

from utils import cached
from .utils import t_test


def VPNL_category_dataset(data_dir=VPNL, frames_per_video=24, video_fps=12):
    """Create a category dataset for the VPNL dataset."""
    datasets = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            category = fname.split('-')[0]
            if category in {'adult', 'child'}:
                category = 'face'
            elif category in {'word', 'number'}:
                category = 'character'
            elif category in {'corridor', 'house'}:
                category = 'place'
            elif category in {'car', 'instrument'}:
                category = 'object'
            elif category in {'body', 'limb'}:
                category = 'body'
            else:
                assert category == 'scrambled'
                continue # skip scrambled images
            datasets[category].append((os.path.join(data_dir, fname), category))
    return datasets

def detailed_VPNL_category_dataset(data_dir=VPNL, frames_per_video=24, video_fps=12):
    """Create a category dataset for the VPNL dataset."""
    datasets = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            category = fname.split('-')[0]
            if category in {'adult', 'child'}:
                category = 'face-detailed'
            elif category in {'word', 'number'}:
                category = 'character-detailed'
            elif category in {'corridor', 'house'}:
                category = 'place-detailed'
            elif category in {'car'}:
                category = 'car-detailed'
            elif category in {'instrument'}:
                category = 'instrument-detailed'
            elif category in {'body', 'limb'}:
                category = 'body-detailed'
            else:
                assert category == 'scrambled'
                continue # skip scrambled images
            datasets[category].append((os.path.join(data_dir, fname), category))
    return datasets

def KANWISHER_category_dataset(data_dir=KANWISHER, frames_per_video=24, video_fps=12):
    """Create a category dataset for the Kanwisher dataset."""
    datasets = defaultdict(list)
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for fname in os.listdir(category_dir):
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                    if category == "Scrambled15G":
                        datasets["Scrambled"].append((os.path.join(category_dir, fname), category))
                    else:
                        datasets[category].append((os.path.join(category_dir, fname), category))
    return datasets


def functional_localization_one_vs_rest(model, transform, datasets, 
                                        batch_size=32, device='cuda', downsampler=None,
                                        video_fps=12, frames_per_video=24, kanwisher=False):

    # Group files by category
    n_categories = len(datasets)
    if kanwisher:
        categories = ["Faces", "Bodies", "Scenes", "Objects", "Scrambled"]
        datasets = {cat: datasets[cat] for cat in categories}
        contrasts = [
            (1, 0, 0, -1, 0),  # Faces vs Objects
            (0, 1, 0, -1, 0),  # Bodies vs Objects
            (0, 0, 1, -1, 0),  # Scenes vs Objects
            (0, 0, 0, 1, -1),  # Objects vs Scrambled
        ]

    else:
        contrasts = [[1 if i == j else -1 for i in range(n_categories)] for j in range(n_categories)]

    t_vals_dict, p_vals_dict = t_test(
        model, transform, datasets=datasets,
        contrasts=contrasts,
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )

    # lahner 2024
    if kanwisher:
        t_val_ret = {}
        t_val_ret["Faces_localizer"] = t_vals_dict["Faces_vs_Objects"]
        t_val_ret["Bodies_localizer"] = t_vals_dict["Bodies_vs_Objects"]
        t_val_ret["Scenes_localizer"] = t_vals_dict["Scenes_vs_Objects"]
        t_val_ret["Objects_localizer"] = t_vals_dict["Objects_vs_Scrambled"]
    else:
        t_val_ret = {cat.split("_vs_")[0]: t_vals for cat, t_vals in t_vals_dict.items()}

    if kanwisher:
        p_vals_ret = {}
        p_vals_ret["Faces_localizer"] = p_vals_dict["Faces_vs_Objects"]
        p_vals_ret["Bodies_localizer"] = p_vals_dict["Bodies_vs_Objects"]
        p_vals_ret["Scenes_localizer"] = p_vals_dict["Scenes_vs_Objects"]
        p_vals_ret["Objects_localizer"] = p_vals_dict["Objects_vs_Scrambled"]
    else:
        p_vals_ret = {cat.split("_vs_")[0]: p_vals for cat, p_vals in p_vals_dict.items()}

    return t_val_ret, p_vals_ret


def localize_categories(model, transform, dataset_name, frames_per_video=24, video_fps=12, 
                        batch_size=32, device='cuda', downsampler=None, ret_pvals=False):
    if dataset_name == "vpnl":
        datasets = VPNL_category_dataset(frames_per_video=frames_per_video, video_fps=video_fps)
    elif dataset_name == "vpnl_detailed":
        datasets = detailed_VPNL_category_dataset(frames_per_video=frames_per_video, video_fps=video_fps)
    elif dataset_name == "kanwisher":
        datasets = KANWISHER_category_dataset(frames_per_video=frames_per_video, video_fps=video_fps)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    t_vals_dict, p_vals_dict = functional_localization_one_vs_rest(
        model, transform=transform, datasets=datasets,
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video, kanwisher=(dataset_name=="kanwisher")
    )
    if ret_pvals:
        return t_vals_dict, p_vals_dict
    return t_vals_dict