from config import ROOT_AFD101
import os
from typing import Tuple, Optional, List

from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import ConcatDataset

# Assume these exist in utils module
from .utils import video_from_imgs
from .utils.transforms import Transforms
from .utils.io import Video


class AFD101:
    def __init__(
        self,
        root: str = ROOT_AFD101,
        fps: int = 12,
        duration: int = 2000,
        size: Tuple[int, int] = (224, 224),
        fold: Optional[int] = 1,
        test_transforms: Optional[List[str]] = None,
    ):
        """
        Args:
            root: Root directory containing 'videos' and 'splits' subdirectories
            fps: Frames per second
            duration: Video duration in milliseconds
            size: (width, height) tuple for video frames
            fold: Fold number (1-3) for cross-validation
            test_transforms: List of transform names to apply
        """
        videos_root = os.path.join(root, 'videos')
        annotation_path = os.path.join(root, 'splits')
        
        if fold is None:
            fold = 1
        
        val_transforms = test_transforms

        trainset = _AFD101(
            videos_root,
            annotation_path,
            val_transforms,
            train=True,
            fold=fold,
            fps=fps,
            duration=duration,
            size=size
        )
        valset = _AFD101(
            videos_root,
            annotation_path,
            val_transforms,
            train=False,
            fold=fold,
            fps=fps,
            duration=duration,
            size=size
        )

        self.trainset = trainset
        self.valset = ConcatDataset([trainset, valset])

    @property
    def num_classes(self):
        return self.trainset.num_classes


class _AFD101(VisionDataset):
    def __init__(
        self,
        root: str,
        annotation_path: str,
        transforms,
        train: bool,
        fold: int = 1,
        fps: int = 12,
        duration: int = 2000,
        size: Tuple[int, int] = (224, 224)
    ):
        super(_AFD101, self).__init__(root=root, transforms=transforms)
        
        if not 1 <= fold <= 3:
            raise ValueError(f"fold should be between 1 and 3, got {fold}")
        
        self.dataset_name = 'afd101'
        self.root = root
        
        extensions = ('avi',)
        self.fold = fold
        self.fps = fps
        self.duration = duration
        self.size = size
        self.train = train
        self.transforms = transforms

        classes = list(sorted(list_dir(root)))
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.samples = make_dataset(
            root, self.class_to_idx, extensions, is_valid_file=None
        )
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = [video_list[i] for i in self.indices]

    def _select_fold(
        self,
        video_list: List[str],
        annotation_path: str,
        fold: int,
        train: bool
    ) -> List[int]:
        """Select video indices based on fold and train/test split."""
        name = "train" if train else "test"
        name = f"{name}list{fold:02d}.txt"
        f = os.path.join(annotation_path, name)
        
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [x[0] for x in data]
            selected_files.extend(data)
        
        selected_files = set(selected_files)
        indices = [
            i for i in range(len(video_list))
            if video_list[i][len(self.root) + 1:] in selected_files
        ]
        return indices

    def __getitem__(self, idx: int) -> Tuple:
        """Get video data and label."""
        video_path, label = self.samples[self.indices[idx]]

        if self.transforms is None:
            return video_path, label, self.dataset_name
        
        video = Video.from_path(video_path)
        video = video.set_fps(self.fps)
        video = video.set_window(0, self.duration).set_size(self.size)
        imgs = video.to_tensor(cache=True)
        
        data = video_from_imgs(imgs, self.transforms)

        data = data.permute(1, 0, 2, 3)  # TCHW
        
        return data, label, self.dataset_name
    
    def __len__(self) -> int:
        return len(self.video_clips)
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)
