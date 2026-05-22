import random

import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T
# from pytorchvideo.transforms import AugMix, Div255

class Transforms:
    def __init__(self, size):
        self.templates = {
            # Commented is imagenet, uncommented is s3d on kinetics400
            # 'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 'resize': T.Compose([
            #     T.Resize((256, 256), interpolation=IM.BILINEAR),
            #     T.CenterCrop((224, 224))
            # ]),
            'standardize': Div255(),
            'normalize': T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            'normalize-vit': T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            'crop': T.CenterCrop(size),
            'resize': T.Resize(size),
            'augmix': AugMix() # TODO might need to play around magnitude and width
        }

    def get_transforms(self, lst):
        return T.Compose(
            [self.templates[key] for key in lst]
        )
    
def shake(data, max_translation):
    C, T, H, W = data.shape
        
    dx = torch.randint(-max_translation, max_translation + 1, (T,), device=data.device)
    dy = torch.randint(-max_translation, max_translation + 1, (T,), device=data.device)

    shaken_frames = [
        F.affine(data[:, t], angle=0, translate=(dx[t].item(), dy[t].item()), scale=1.0, shear=0,
                 interpolation=F.InterpolationMode.BILINEAR)
        for t in range(T)
    ]
    
    return torch.stack(shaken_frames, dim=1) # Stack along T dimension