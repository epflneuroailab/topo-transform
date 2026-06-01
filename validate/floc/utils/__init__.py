import torch
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy import stats

import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .basic import visualize_all_rois
from .basic import visualize_all_rois_v2
from .basic import visualize_tvals
from .cluster import find_patches
from .cluster import find_patches_for_categories
from .cluster import labels_to_unit_indices
from .cluster import visualize_patches

from utils import cached


class CategoryDataset(Dataset):
    def __init__(self, file_infos, transform, frames_per_video=24, video_fps=12):
        self.file_paths = [info[0] for info in file_infos]
        self.labels = [info[1] for info in file_infos]
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.video_fps = video_fps
        self.video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        # Extract unique categories from labels
        self.categories = sorted(list(set(self.labels)))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
    
    def __len__(self):
        return len(self.file_paths)
    
    def _is_video(self, path):
        return Path(path).suffix.lower() in self.video_exts
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.category_to_idx[self.labels[idx]]
        
        if self._is_video(file_path):
            decoder = safe_decoder(file_path)
            duration = decoder.metadata.duration_seconds
            time_duration = self.frames_per_video / self.video_fps
            
            if duration < time_duration:
                time_start, time_end = 0, time_duration
            else:
                time_start = (duration - time_duration) / 2
                time_end = time_start + time_duration

            data = video_transform(
                file_path, time_start, time_end,
                torch_transforms=self.transform,
                fps=self.video_fps
            )
        else:
            img = torch.from_numpy(np.array(Image.open(file_path).convert('RGB'))).permute(2,0,1)
            img = self.transform(img)
            data = img.unsqueeze(0).repeat(self.frames_per_video, 1, 1, 1)
        
        return (data, label)


def run_features(model, data_loader, device, downsampler=None):
    """Run feature extraction on a data loader.
    
    Returns:
        offline_feats: np.ndarray or list of np.ndarray [num_samples, ...]
                       assuming features [B, T, ...]
        offline_targets: np.ndarray [num_samples, ...]
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    all_feats = []
    all_targets = []

    for tmp in tqdm(data_loader, desc="Extracting features"):
        data, target = tmp[0], tmp[1]
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(data)[0]
            
            # Apply downsampler if provided
            if downsampler is not None:
                outputs = [downsampler(out) for out in outputs] if isinstance(outputs, (list, tuple)) else downsampler(outputs)
            
            # Handle list or single output
            if isinstance(outputs, (list, tuple)):
                outputs = [out.cpu().numpy() for out in outputs]
                if len(all_feats) == 0:
                    all_feats = [[] for _ in range(len(outputs))]
                for j, out in enumerate(outputs):
                    all_feats[j].append(out)
            else:
                all_feats.append(outputs.cpu().numpy())
        
        all_targets.append(target.cpu().numpy())

    # Concatenate results
    if isinstance(all_feats[0], list):
        all_feats = [np.concatenate(feats, 0) for feats in all_feats]
    else:
        all_feats = np.concatenate(all_feats, 0)
    all_targets = np.concatenate(all_targets, 0)

    return all_feats, all_targets


def safe_decoder(path):
    """Create a VideoDecoder safely"""
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(path)


def safe_get_frames(decoder, frame_indices, max_attempt=10):
    """
    Safely get frames from a torchcodec decoder.

    - If frame 0 fails, searches for the first valid frame up to max_attempt.
    - Shifts all indices so decoding starts from the first valid frame.
    """
    # Try to decode requested frames
    try:
        return decoder.get_frames_at(frame_indices).data
    except RuntimeError as e:
        print(f"[WARN] Failed to get frames {frame_indices}: {e}")
        # search for first valid frame
        first_valid = None
        for i in range(max_attempt):
            try:
                _ = decoder.get_frames_at([i]).data
                first_valid = i
                print(f"[INFO] Found first valid frame at index {i}")
                break
            except RuntimeError:
                continue
        if first_valid is None:
            raise RuntimeError(f"No valid frames found in first {max_attempt} frames for {decoder.source}")

        # shift indices to start from first_valid
        safe_indices = [max(idx, first_valid) for idx in frame_indices]
        return decoder.get_frames_at(safe_indices).data


def video_transform(path, time_start, time_end, torch_transforms, fps=12):
    """
    Load video, clip by time, pad out-of-bound frames with gray [255/2] frames,
    resample fps, and apply torchvision transforms.

    Args:
        path (str): path to video file
        time_start (float): start time in seconds
        time_end (float): end time in seconds
        torch_transforms (callable): torchvision transforms to apply
        fps (int): target frames per second

    Returns:
        Tensor: [num_frames, channels, height, width]
    """
    decoder = safe_decoder(path)
    metadata = decoder.metadata
    num_video_frames = metadata.num_frames
    video_fps = metadata.average_fps
    video_duration = metadata.duration_seconds

    # Convert requested times to frame indices (float)
    start_frame_f = time_start * video_fps
    end_frame_f = time_end * video_fps

    # Compute actual frame indices in bounds
    start_idx = max(0, int(start_frame_f))
    end_idx = min(num_video_frames, int(end_frame_f))

    # Number of frames before/after for padding
    pad_start = max(0, int(-start_frame_f))
    pad_end = max(0, int(end_frame_f - num_video_frames))

    # Extract frames in bounds
    if start_idx < end_idx:
        frame_indices = list(range(start_idx, end_idx))
        frames = safe_get_frames(decoder, frame_indices)  # [num_frames, C, H, W]
    else:
        frames = torch.empty((0, 0, 0, 0))

    # Determine target number of frames
    num_target_frames = int(round((time_end - time_start) * fps))

    # Case 1: Entire range is outside video → return pure gray frames
    if start_idx >= end_idx:
        frames = torch.full(
            (num_target_frames, 3, 224, 224), 255 / 2, dtype=torch.float32
        )
    else:
        # Pad start
        if pad_start > 0:
            pad_frames = torch.full(
                (pad_start, frames.shape[1], frames.shape[2], frames.shape[3]),
                255 / 2,
                dtype=frames.dtype,
            )
            frames = torch.cat([pad_frames, frames], dim=0)

        # Pad end
        if pad_end > 0:
            pad_frames = torch.full(
                (pad_end, frames.shape[1], frames.shape[2], frames.shape[3]),
                255 / 2,
                dtype=frames.dtype,
            )
            frames = torch.cat([frames, pad_frames], dim=0)

        # Resample to target FPS
        if frames.shape[0] != num_target_frames and frames.shape[0] > 0:
            indices = torch.linspace(0, frames.shape[0] - 1, steps=num_target_frames).long()
            frames = frames[indices]

    if frames.numel() > 0:
        # apply transforms
        frames = torch_transforms(frames)

    return frames

def _t_test(model, transform, datasets, contrasts, batch_size=32, device='cuda', downsampler=None, video_fps=12, frames_per_video=24, related=False):
    # datasets: map(category: str, List[(file_path: str, label: str)])
    #           or map(category: str, dataset: Dataset) 
    categories = list(datasets.keys())
    print(f"Found categories: {categories}")
    print(f"Activations over time are averaged for each stimulus.")
    
    # ensure contrasts are valid
    for contrast in contrasts:
        for c in contrast:
            assert isinstance(c, int)
            assert c in [0, 1, -1], f"Invalid contrast category: {c}"

    # Extract features
    category_feats = {}
    for category, file_infos in datasets.items():
        print(f"Extracting features for {category}...")
        if isinstance(file_infos, Dataset):
            cat_dataset = file_infos
        else:
            cat_dataset = CategoryDataset(file_infos, transform=transform, 
                                        frames_per_video=frames_per_video,
                                        video_fps=video_fps)
        loader = DataLoader(cat_dataset, batch_size=batch_size, 
                            num_workers=int(batch_size/1.5), shuffle=False)   # Important: don't shuffle to maintain correspondence
        feats, _ = run_features(model, loader, device, downsampler)
        feats = [np.mean(f, axis=1) for f in feats]  # NOTE: average over time
        category_feats[category] = feats
    
    # Compute one-vs-rest t-statistics
    t_vals_dict = defaultdict(list)
    p_vals_dict = defaultdict(list)
    for contrast in contrasts:
        positive_cats = [categories[i] for i, c in enumerate(contrast) if c == 1]
        negative_cats = [categories[i] for i, c in enumerate(contrast) if c == -1]
        
        # Check if there are no negative values (i.e., testing positive categories against baseline)
        if len(negative_cats) == 0:
            # One-sample t-test against 0
            contrast_name = "_vs_baseline".join(["+".join(positive_cats), ""])
            print(f"Computing t-stats for {contrast_name} (one-sample t-test against 0)...")
            
            num_layers = len(category_feats[positive_cats[0]])
            positive_feats = [
                np.concatenate([category_feats[cat][i] for cat in positive_cats], axis=0)
                for i in range(num_layers)
            ]
            
            # Use one-sample t-test against 0
            for i in range(num_layers):
                ts, ps = stats.ttest_1samp(positive_feats[i], 0, axis=0)
                t_vals_dict[contrast_name].append(ts)
                p_vals_dict[contrast_name].append(ps)
        else:
            # Two-sample t-test (original code)
            contrast_name = "_vs_".join(["+".join(positive_cats), "+".join(negative_cats)])
            print(f"Computing t-stats for {contrast_name}...")

            num_layers = len(category_feats[positive_cats[0]])
            positive_feats = [
                np.concatenate([category_feats[cat][i] for cat in positive_cats], axis=0)
                for i in range(num_layers)
            ]
            negative_feats = [
                np.concatenate([category_feats[cat][i] for cat in negative_cats], axis=0)
                for i in range(num_layers)
            ]
            ttest_alg = stats.ttest_ind if not related else stats.ttest_rel
            for i in range(num_layers):
                ts, ps = ttest_alg(
                    positive_feats[i], negative_feats[i], axis=0
                )
                t_vals_dict[contrast_name].append(ts)
                p_vals_dict[contrast_name].append(ps)

        # Print summary statistics
        for i in range(num_layers):
            print(f"  Layer {i}: mean t = {np.mean((t_vals_dict[contrast_name][i])):.3f}, "
                  f"max |t| = {np.max(np.abs(t_vals_dict[contrast_name][i])):.3f}")
                  
    return t_vals_dict, p_vals_dict

def t_test(model, transform, datasets, contrasts, related=False, **kwargs):
    """Wrapper function for t_test with caching."""

    contrast_hash = hashlib.md5(str(contrasts).encode()).hexdigest()[:8]
    datasets_hash = hashlib.md5(str(sorted(datasets.keys())).encode()).hexdigest()[:8]
    cache_key = (
        f'ttest{"_rel" if related else ""}_{model.name}_'
        f'{datasets_hash}_'
        f'{contrast_hash}'
    )

    if model.smoothing:
        cache_key += f'_smoothed_fwhm{model.fwhm_mm}_res{model.resolution_mm}'

    t_vals_dict, p_vals_dict = cached(cache_key)(_t_test)(model, transform, datasets, contrasts, related=related, **kwargs)
    
    t_vals_dict = dict(t_vals_dict)
    p_vals_dict = dict(p_vals_dict)

    return t_vals_dict, p_vals_dict


__all__ = [
    "CategoryDataset",
    "find_patches",
    "find_patches_for_categories",
    "labels_to_unit_indices",
    "run_features",
    "safe_decoder",
    "safe_get_frames",
    "t_test",
    "video_transform",
    "visualize_all_rois",
    "visualize_all_rois_v2",
    "visualize_patches",
    "visualize_tvals",
]
