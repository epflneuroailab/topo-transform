from config import ROOT_AFRAZ2006
import os
from typing import Tuple, Optional, List
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import ConcatDataset

# Assume these exist in utils module
from .utils import video_from_imgs
from .utils.transforms import Transforms
from .utils.io import Video


class AFRAZ2006:
    def __init__(
        self,
        root: str = ROOT_AFRAZ2006,
        fps: int = 12,
        duration: int = 2000,
        size: Tuple[int, int] = (224, 224),
        transforms: Optional[List[str]] = None,
    ):
        """
        Args:
            root: Root directory containing 'videos' and 'splits' subdirectories
            fps: Frames per second
            duration: Video duration in milliseconds
            size: (width, height) tuple for video frames
        """

        images_root_train = os.path.join(root, 'train')
        images_root_test = os.path.join(root, 'test')
        annotation_path_train = os.path.join(root, 'stimulus_Afraz2006_train.csv')
        annotation_path_test = os.path.join(root, 'stimulus_Afraz2006_test.csv')
        
        trainset = _AFRAZ2006(
            images_root_train,
            annotation_path_train,
            transforms,
            fps=fps,
            duration=duration,
            size=size
        )
        valset = _AFRAZ2006(
            images_root_test,
            annotation_path_test,
            transforms,
            fps=fps,
            duration=duration,
            size=size
        )

        self.trainset = trainset
        self.valset = valset

    @property
    def num_classes(self):
        return self.trainset.num_classes


class _AFRAZ2006(VisionDataset):
    def __init__(
        self,
        root: str,
        annotation_path: str,
        transforms,
        fps: int = 12,
        duration: int = 2000,
        size: Tuple[int, int] = (224, 224)
    ):
        super(_AFRAZ2006, self).__init__(root=root, transforms=transforms)
        
        self.dataset_name = 'afraz2006'
        self.root = root
        
        self.fps = fps
        self.duration = duration
        self.size = size
        self.transforms = transforms
        self.annot = pd.read_csv(annotation_path)
        self.samples = []
        for idx, row in self.annot.iterrows():
            filename = row['filename']
            label_signal_level = row['label_signal_level']
            label = int(row['image_label'] == "face")  # label: 1-face
            img_path = os.path.join(self.root, filename)
            self.samples.append((img_path, label, label_signal_level))

    def label_signal_levels(self):
        """Get unique noise levels in the dataset."""
        return [d[2] for d in self.samples]

    def __getitem__(self, idx: int) -> Tuple:
        """Get video data and label."""
        img_path, label, label_signal_level = self.samples[idx]

        if self.transforms is None:
            return img_path, label, self.dataset_name
        
        frames_per_video = int(self.fps * (self.duration / 1000))
        img = torch.from_numpy(np.array(Image.open(img_path).convert('RGB'))).permute(2,0,1)
        img = self.transforms(img)
        data = img.unsqueeze(0).repeat(frames_per_video, 1, 1, 1)
        
        return data, label, self.dataset_name
    
    def __len__(self) -> int:
        return len(self.samples)
    
    @property
    def num_classes(self) -> int:
        return 2


if __name__ == "__main__":
    dataset = AFRAZ2006(transforms=None)
    print(f"Train set size: {len(dataset.trainset)}")
