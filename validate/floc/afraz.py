import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import copy

from utils import cached
from data import AFRAZ2006
from .utils import t_test


# Produce motion index by reporting t-values contrasting Kinetics400 vs imagenet

def face_selectivity(model, transform, dataset, 
                 batch_size=32, device='cuda', downsampler=None, frames_per_video=24, video_fps=12):

    datasets = {
        "face": [],
        "nonface": [],
    }


    for idx in range(len(dataset)):
        file_path, label, _ = dataset[idx]
        if label == 1:
            datasets['face'].append((file_path, 'face'))
        else:
            datasets['nonface'].append((file_path, 'nonface'))

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


def localize_afraz(model, transform, frames_per_video=24, video_fps=12,
                    batch_size=32, device='cuda', num_samples=256, seed=42, ret_pvals=False):

    dataset = AFRAZ2006().trainset

    t_vals_dict, p_vals_dict = face_selectivity(
        model, 
        transform, 
        dataset, 
        batch_size=batch_size, 
        device=device,
    )

    if ret_pvals:
        return t_vals_dict, p_vals_dict
    return t_vals_dict