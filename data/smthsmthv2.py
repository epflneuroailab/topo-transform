from config import ROOT_SSV2
import os
import json

import numpy as np
import torch

from torch.utils.data import Dataset

from .utils.transforms import Transforms
from .utils import ListData, video_from_imgs
from .utils.io import Video

class SmthSmthV2():
    def __init__(self, root=ROOT_SSV2, fps=12, duration=2000, width=224, height=224, 
                 train_transforms=None, test_transforms=None, shuffle=False, 
                 static=False, debug=False, subsample_factor=0.1):
        videos_root = root + '/videos'
        json_file_train = root + '/annotations/something-something-v2-train.json'
        json_file_val = root + '/annotations/something-something-v2-validation.json'
        json_file_labels = root + '/annotations/something-something-v2-labels.json'
        
        size = (width, height)

        self.trainset = _SmthSmthV2(
            videos_root, json_file_train, json_file_labels, train=True,
            fps=fps, duration=duration, size=size, transforms=train_transforms, 
            shuffle=shuffle, static=static, subsample_factor=subsample_factor
        )
        self.valset = _SmthSmthV2(
            videos_root, json_file_val, json_file_labels, train=False,
            fps=fps, duration=duration, size=size, transforms=test_transforms, 
            shuffle=shuffle, static=static, subsample_factor=0.02
        )

        if not debug:
            assert self.trainset.num_classes == self.valset.num_classes == 174

    @property
    def num_classes(self):
        return self.trainset.num_classes

class _SmthSmthV2(Dataset):
    def __init__(self, root, json_file_input, json_file_labels, train, transforms, fps=12,
                 duration=2000, size=(224, 224), shuffle=False, static=False, subsample_factor=1.0):
        self.dataset_name = 'smthsmthv2'

        self.json_file_input = json_file_input
        self.json_file_labels = json_file_labels
        self.root = root
        self.fps = fps
        self.duration = duration
        self.size = size
        self.train = train
        self.transforms = transforms
        self.shuffle = shuffle
        self.static = static
        self.subsample_factor = subsample_factor
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()

    def read_json_labels(self):
        classes = []
        with open(self.json_file_labels, 'rb') as json_file:
            json_reader = json.load(json_file)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)
    
    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict

    def read_json_input(self):
        json_data = []
        with open(self.json_file_input, 'rb') as json_file:
            json_reader = json.load(json_file)
            for elem in json_reader:
                label = self.clean_template(elem['template'])
                if label not in self.classes:
                    raise ValueError(f'Label {label} not found in classes')
                item = ListData(elem['id'], label,
                                os.path.join(self.root, elem['id'] + '.webm'))
                json_data.append(item)

        # --------------------------
        # CLASS-BALANCED SUBSAMPLING
        # --------------------------
        if self.subsample_factor < 1.0:
            self.num_classes_limit = 50
            from collections import defaultdict

            # 1. Group by class
            class_buckets = defaultdict(list)
            for item in json_data:
                class_buckets[item.label].append(item)

            # 2. Restrict the number of classes to 50
            # -------------------------------------------------
            max_classes = 50
            all_classes = list(class_buckets.keys())
            all_classes.sort()

            if len(all_classes) > max_classes:
                np.random.seed(42)
                selected_classes = np.random.choice(all_classes, max_classes, replace=False)
                selected_classes = set(selected_classes)  # for fast lookup
            else:
                selected_classes = set(all_classes)

            # Keep only selected classes
            class_buckets = {cls: items for cls, items in class_buckets.items() if cls in selected_classes}

            # 3. Class-balanced subsampling
            # -------------------------------------------------
            balanced = []
            for cls, items in class_buckets.items():
                n = len(items)
                k = max(1, int(np.ceil(n * self.subsample_factor)))
                idx = np.random.choice(n, k, replace=False)
                balanced.extend([items[i] for i in idx])

            json_data = balanced

        return json_data
    
    def clean_template(self, template):
        return template.replace('[', '').replace(']', '')
        
    def __getitem__(self, index):
        try:
            item = self.json_data[index]
            video = Video.from_path(item.path)

        except Exception as e:
            print(f"Error processing video {item.path}: {e}")
            return self.__getitem__((index + 1) % len(self))

        if self.transforms is None:
            return item.path, self.classes_dict[item.label], self.dataset_name

        video = video.set_fps(self.fps)
        
        # randomly select a segment
        # I think the problem is that the duration is badly guessed by cv2, hence the last invalid
        # frames (they don't exist although opencv thinks they do). By reducing the total_duration
        # by 1s, we limit the risk of going over and having too many invalid frames
        # Since there are still videos for which it breaks, I added a temporary try/except
        # to at least be able to make it run...
        total_duration = max(0, video.duration - 1000)
        start = max(0, np.random.rand() * (total_duration - self.duration))
        video = video.set_window(start, start + self.duration)

        imgs = video.to_tensor()

        label = item.label
        label = self.classes_dict[label]

        data = video_from_imgs(imgs, self.transforms)
        if self.shuffle and not self.static:
            idx = torch.randperm(data.shape[1]) # shuffle frames
            data = data[:, idx, :, :]
        if self.static:
            idx = torch.randperm(data.shape[1])[0]
            data = data[:, idx:idx+1, :, :].repeat(1, data.shape[1], 1, 1)

        data = data.permute(1, 0, 2, 3)  # TCHW

        return data, label, self.dataset_name
    
    def __len__(self):
        return len(self.json_data)
    
    @property
    def num_classes(self):
        return len(self.classes)