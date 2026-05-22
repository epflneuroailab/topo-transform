from config import ROOT_IMAGENETVID
import os
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset
import numpy as np

# Assume these exist in utils module
from .utils import video_from_imgs, Image
from .utils.transforms import Transforms, shake


class ImageNetVid:
    def __init__(
        self,
        root: str = ROOT_IMAGENETVID,
        fps: int = 12,
        duration: int = 2000,
        size: Tuple[int, int] = (224, 224),
        shake_intensity: int = 0,
        train_transforms: Optional[List[str]] = None,
        test_transforms: Optional[List[str]] = None,
        debug: bool = False,
        subsample_factor: float = 0.01,
    ):
        """
        Args:
            root: Root directory containing 'train' and 'val' subdirectories
            fps: Frames per second
            duration: Video duration in milliseconds
            size: (width, height) tuple for frames
            shake_intensity: Intensity of shake augmentation (0 = no shake)
            train_transforms: List of transform names for training
            test_transforms: List of transform names for validation
            debug: If True, skip assertion checks
        """
        videos_root = root
        
        _train_transforms = train_transforms
        _val_transforms = test_transforms

        self.trainset = _ImageNetVid(
            videos_root,
            _train_transforms,
            fps=fps,
            duration=duration,
            size=size,
            split='train',
            shake=shake_intensity,
            subsample_factor=subsample_factor
        )
        self.valset = _ImageNetVid(
            videos_root,
            _val_transforms,
            fps=fps,
            duration=duration,
            size=size,
            split='val',
            shake=shake_intensity,
            subsample_factor=0.02
        )

    @property
    def num_classes(self):
        return self.trainset.num_classes


class _ImageNetVid(Dataset):
    def __init__(
        self,
        root: str,
        transforms,
        fps: int = 12,
        duration: int = 2000,
        size: Tuple[int, int] = (224, 224),
        split: str = 'train',
        shake: int = 0,
        subsample_factor: float = 1.0,
    ):
        """
        Args:
            root: Root directory
            transforms: Transforms to apply
            fps: Frames per second
            duration: Video duration in milliseconds
            size: (width, height) tuple for frames
            split: 'train' or 'val'
            shake: Shake augmentation intensity
        """
        self.dataset_name = 'imagenet'
        
        self.root = os.path.join(root, split)
        self.transforms = transforms
        self.fps = fps
        self.duration = duration
        self.size = size
        self.shake = shake

        self.image_paths = []
        self.labels = []

        classes = sorted(os.listdir(self.root))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        # Collect all image paths and labels
        for class_name in classes:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])

        if subsample_factor < 1.0:
            self.num_classes_limit = 100
            from collections import defaultdict

            # group indices by class
            per_class = defaultdict(list)
            for idx, lbl in enumerate(self.labels):
                per_class[lbl].append(idx)

            # ---------------------------
            # restrict number of classes
            # ---------------------------
            max_classes = self.num_classes_limit
            all_classes = list(per_class.keys())
            all_classes.sort()
            if len(all_classes) > max_classes:
                np.random.seed(42)
                selected_classes = np.random.choice(all_classes, max_classes, replace=False)
                selected_classes = set(selected_classes)
            else:
                selected_classes = set(all_classes)

            # filter per_class to only selected classes
            per_class = {lbl: idxs for lbl, idxs in per_class.items() if lbl in selected_classes}

            # ---------------------------
            # class-balanced subsampling
            # ---------------------------
            selected_indices = []
            for lbl, idxs in per_class.items():
                k = max(1, int(len(idxs) * subsample_factor))  # ensure at least one sample per class
                chosen = np.random.choice(idxs, k, replace=False)
                selected_indices.extend(chosen)

            # sort to keep order stable (optional)
            selected_indices = sorted(selected_indices)

            # apply subsampling
            self.image_paths = [self.image_paths[i] for i in selected_indices]
            self.labels = [self.labels[i] for i in selected_indices]

    def __getitem__(self, idx: int) -> Tuple:
        """Get video data from static image."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        if self.transforms is None:
            return img_path, label, self.dataset_name
        
        # Load and process image
        image = Image.from_path(img_path).set_size(self.size)
        frame = torch.from_numpy(image.to_numpy())
        frames = frame[None, :]
        frames = video_from_imgs(frames, self.transforms)
        
        # Expand single frame to video
        num_frames = self.fps * self.duration // 1000
        data = frames.expand(-1, num_frames, -1, -1)
        
        # Apply shake augmentation if needed
        if self.shake > 0:
            data = shake(data, self.shake)

        data = data.permute(1, 0, 2, 3)  # TCHW

        return data, label, self.dataset_name

    def __len__(self) -> int:
        return len(self.image_paths)
    
    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)
