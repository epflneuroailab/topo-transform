from config import ROOT_KINETICS400

import os
import numpy as np
import glob
from typing import Optional, Tuple, List, Callable

import torch
from torch.utils.data import Dataset, Subset

from .utils.transforms import Transforms
from .utils import ListData, video_from_imgs
from .utils.io import Video

class Kinetics400:
    def __init__(
        self,
        root: str = ROOT_KINETICS400,
        fps: int = 12,
        duration: int = 2000,
        size: Tuple[int, int] = (224, 224),
        train_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        shuffle: bool = False,
        static: bool = False,
        debug: bool = False,
        subsample_factor: float = 0.1,
    ):
        """
        Kinetics400 dataset wrapper.
        
        Args:
            root: Root directory containing 'videos/train' and 'videos/val'
            fps: Frames per second for video sampling
            duration: Duration in milliseconds to sample from each video
            size: (width, height) for video frames
            train_transforms: Transforms to apply to training data
            test_transforms: Transforms to apply to validation data
            shuffle: Whether to shuffle frames temporally
            static: Whether to repeat a single random frame
            debug: If False, asserts that both splits have 400 classes
        """
        self.trainset = _Kinetics400(
            root, train=True, transforms=train_transforms,
            fps=fps, duration=duration, size=size, shuffle=shuffle, static=static,
            subsample_factor=subsample_factor
        )
        self.valset = _Kinetics400(
            root, train=False, transforms=test_transforms,
            fps=fps, duration=duration, size=size, shuffle=shuffle, static=static,
            subsample_factor=0.02
        )

    @property
    def num_classes(self) -> int:
        return self.trainset.num_classes


class _Kinetics400(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        transforms: Optional[Callable] = None,
        fps: int = 12,
        duration: int = 2000,
        size: Tuple[int, int] = (224, 224),
        shuffle: bool = False,
        static: bool = False,
        subsample_factor: float = 1.0,
    ):
        """
        Internal Kinetics400 dataset implementation.
        
        Args:
            root: Root directory containing 'videos' folder
            train: Whether this is the training split
            transforms: Transforms to apply to video frames
            fps: Frames per second for video sampling
            duration: Duration in milliseconds to sample from each video
            size: (width, height) for video frames
            shuffle: Whether to shuffle frames temporally
            static: Whether to repeat a single random frame
        """
        self.dataset_name = 'kinetics400'

        split = "train" if train else "val"
        self.root = os.path.join(root, split)
        self.transforms = transforms
        self.fps = fps
        self.duration = duration
        self.size = size
        self.shuffle = shuffle
        self.static = static

        print(self.root)

        self.video_paths = glob.glob(os.path.join(self.root, "*.mp4"))
        self.classes = self._load_classes()

        # randomly subsample the dataset if subsample_factor < 1.0
        if subsample_factor < 1.0:
            num_videos = len(self.video_paths)
            subsample_size = int(num_videos * subsample_factor)
            self.video_paths = list(np.random.choice(self.video_paths, subsample_size))

    def _load_classes(self) -> dict:
        """Load class names and create class-to-index mapping."""
        class_names = sorted(set(
            os.path.basename(os.path.dirname(path)) 
            for path in self.video_paths
        ))
        return {name: idx for idx, name in enumerate(class_names)}
    
    def _get_class(self, video_path: str) -> int:
        """Get class index from video path."""
        class_name = os.path.basename(os.path.dirname(video_path))
        return self.classes[class_name]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a video sample.
        
        Returns:
            data: Video tensor of shape (C, T, H, W)
            label: Class index
            dataset_name: Name of the dataset
        """
        try:
            video_path = self.video_paths[idx]
            # Assumes utils.io.Video exists with this interface
            video = Video.from_path(video_path)

        except Exception as e:
            print(f"Error loading video at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transforms is None:
            return video_path, self._get_class(video_path), self.dataset_name

        video = video.set_fps(self.fps)

        max_start_time = max(0, video.duration - self.duration)
        start = np.random.uniform(0, max_start_time)
        video = video.set_window(start, start + self.duration)

        imgs = video.to_tensor()

        label = self._get_class(video_path)

        # Assumes utils.video_from_imgs exists with this interface
        data = video_from_imgs(imgs, self.transforms)
        
        if self.shuffle and not self.static:
            idx = torch.randperm(data.shape[1])
            data = data[:, idx, :, :]
        if self.static:
            idx = torch.randperm(data.shape[1])[0]
            data = data[:, idx:idx+1, :, :].repeat(1, data.shape[1], 1, 1)
            
        data = data.permute(1, 0, 2, 3)  # TCHW

        return data, label, self.dataset_name
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)