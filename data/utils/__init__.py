from collections import namedtuple

import numpy as np
import torch
from .io import Video, Image

ListData = namedtuple('ListData', ['id', 'label', 'path'])

def video_from_imgs(imgs, transform):
    if isinstance(imgs, list) or isinstance(imgs, np.ndarray):
        # Shape is (T, H, W, C)
        imgs = np.array(imgs)
        # Shape is (T, H, W, C)
        data = torch.from_numpy(imgs.astype(np.float32))
    else:
        data = imgs.float()
    # Need shape (T, C, H, W) https://discuss.pytorch.org/t/can-transforms-compose-handle-a-batch-of-images/4850/5
    data = data.permute(0, 3, 1, 2)
    data = transform(data)
    # Need shape (C, T, H, W) https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
    data = data.permute(1, 0, 2, 3)
    return data