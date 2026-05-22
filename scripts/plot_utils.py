from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_dir(path):
    if path is None:
        return None
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def savefig(path, **kwargs):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, **kwargs)
    plt.close()
    return output_path


def to_numpy(array):
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)
