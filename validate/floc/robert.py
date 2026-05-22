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

from config import ROBERT, ROBERT_STATS, PLOTS_DIR
from utils import cached
from .utils import t_test


def load_robert_tvals():
    t_vals = []
    for individual in os.listdir(ROBERT_STATS):
        if not individual.endswith('.npy'):
            continue
        t_val = np.load(os.path.join(ROBERT_STATS, individual))
        t_vals.append(t_val)
    t_vals = np.array(t_vals)
    t_vals_mean = t_vals.mean(0)

    absvmax = np.max(np.abs(t_vals_mean))

    # # plot
    # from data.neural_data.collections.tang2025 import get_compilation

    # _, _, _, ceilings = get_compilation(None, return_ceiling=True)
    # t_vals_mean[(ceilings.mean(0) < 0.4)] = np.nan

    # from matplotlib import pyplot as plt
    # from nilearn import datasets, plotting
    # fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    # # Plot Left Hemisphere
    # plotting.plot_surf_stat_map(
    #     surf_mesh=fsaverage.flat_left,
    #     stat_map=-t_vals_mean[:10242],
    #     hemi='left',
    #     # bg_map=fsaverage.sulc_left,
    #     title='Robert t-values (Left Hemisphere)',
    #     view='dorsal',
    #     colorbar=True,
    #     cmap='Spectral',
    #     vmin=-absvmax,
    #     vmax=absvmax,
    # )
    # plt.savefig(PLOTS_DIR / "robert_left.png", dpi=400, transparent=True)
    # plt.close()

    # plotting.plot_surf_stat_map(
    #     surf_mesh=fsaverage.infl_left,
    #     stat_map=-t_vals_mean[:10242],
    #     hemi='left',
    #     # bg_map=fsaverage.sulc_left,
    #     title='Robert t-values (Left Hemisphere)',
    #     view='lateral',
    #     colorbar=True,
    #     cmap='Spectral',
    #     vmin=-absvmax,
    #     vmax=absvmax,
    # )
    # plt.savefig(PLOTS_DIR / "robert_left_brain.png", dpi=400, transparent=True)
    # plt.close()
    # exit()

    return t_vals

def localize_robert_human():
    t_vals = load_robert_tvals()
    return {"robert": t_vals.mean(0)}

def Robert_category_dataset(data_dir=ROBERT):
    """Create a category dataset for the Robert dataset."""
    file_infos = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith('.mat') or fname.endswith('.mp4'): continue
        file_infos['static'].append((os.path.join(data_dir, fname), 'static'))
        file_infos['dynamic'].append((os.path.join(data_dir, fname).replace('.png', '.mp4'), 'dynamic'))

    return file_infos

def localize_robert(model, transform, 
                    batch_size=32, device='cuda', downsampler=None,
                    video_fps=12, frames_per_video=36, ret_pvals=False):

    categories = ["dynamic", "static"]

    # Group files by category
    datasets = Robert_category_dataset()
    datasets = {cat: datasets[cat] for cat in categories}
    n_categories = len(categories)
    t_vals_dict, p_vals_dict = t_test(
        model, transform, 
        datasets=datasets, contrasts=[(1, -1), (1, 0), (0, 1)],
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )

    t_vals_ret = {
        "robert": t_vals_dict["dynamic_vs_static"],
        "robert_motion": t_vals_dict["dynamic_vs_baseline"],
        "robert_static": t_vals_dict["static_vs_baseline"],
    }
    p_vals_ret = {
        "robert": p_vals_dict["dynamic_vs_static"],
        "robert_motion": p_vals_dict["dynamic_vs_baseline"],
        "robert_static": p_vals_dict["static_vs_baseline"],
    }

    if ret_pvals:
        return t_vals_ret, p_vals_ret
    return t_vals_ret

    
