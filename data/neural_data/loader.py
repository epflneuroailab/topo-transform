import torch
import torchvision
from torchvision.io import read_video
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset, Subset
import torchcodec
from torchcodec.decoders import VideoDecoder
import numpy as np


## General

class AssemblyDataset(Dataset):
    def __init__(self, assembly, transform):
        self.assembly = assembly
        self.transform = transform

    def __len__(self):
        return self.assembly.num_presentations

    def __getitem__(self, idx):
        stimulus, target = self.assembly.get_data(idx)
        if self.transform:
            stimulus = image_transform(stimulus['stimulus_path'], self.transform)
        return stimulus, target

def get_data_loader(dataset, batch_size, shuffle=False, num_workers=8):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        # num_workers=0,
        # pin_memory=False,
        # prefetch_factor=None,
    )


## Timeseries

class TemporalAssemblyDataset(AssemblyDataset):
    """Dataset for temporal assemblies (e.g., video stimuli and fMRI responses over time)."""

    def __init__(self, assembly, fps, transform, context_duration=2000, response_delay_duration=0, time_bin_stride=1):
        """Initialize TemporalAssemblyDataset.
            - `fps`: Frame rate for video processing
            - `context_duration`: Context window duration (ms)
            - `response_delay_duration`: Response delay (ms)
            - `time_bin_stride`: Temporal stride for sampling
        """

        super().__init__(assembly, transform)
        self.fps = fps
        self.context_duration = context_duration
        self.response_delay_duration = response_delay_duration
        self.time_bin_stride = time_bin_stride

    def __len__(self):
        return self.assembly.num_presentations * (self.assembly.num_time_bins // self.time_bin_stride)

    def __getitem__(self, idx):
        presentation_idx = idx // (self.assembly.num_time_bins // self.time_bin_stride)
        time_bin_idx = (idx % (self.assembly.num_time_bins // self.time_bin_stride)) * self.time_bin_stride
        stimulus, target = self.assembly.get_data(presentation_idx, time_slice=slice(time_bin_idx, time_bin_idx+1))
        time_end = stimulus['time_end'] - self.response_delay_duration
        time_start = time_end - self.context_duration
        if self.transform:
            stimulus = video_transform(
                stimulus['stimulus_path'],
                time_start=time_start / 1000,  # ms to s
                time_end=time_end / 1000,  # ms to s
                torch_transforms=self.transform,
                fps=self.fps
            )
        return stimulus, target

    def subset(self, ratios, time_bin_block=1, time_bin_limit=None, seed=None):
        time_bin_block = max(1, int(np.ceil(time_bin_block / self.time_bin_stride)))
        segments = []
        
        num_time_bins = self.assembly.num_time_bins if time_bin_limit is None else min(self.assembly.num_time_bins, time_bin_limit)
        num_time_bins_strided = num_time_bins // self.time_bin_stride
        
        for pres in range(self.assembly.num_presentations):
            for start in range(0, num_time_bins_strided, time_bin_block):
                end = min(start + time_bin_block, num_time_bins_strided)
                if end - start < time_bin_block and start > 0:
                    start = max(0, end - time_bin_block)
                # Convert to dataset indices: pres * num_time_bins_strided + time_bin_idx
                dataset_indices = [pres * num_time_bins_strided + i for i in range(start, end)]
                segments.append(dataset_indices)
        
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(segments)
        num_segments = len(segments)
        splits = []
        start_idx = 0
        for ratio in ratios:
            end_idx = start_idx + int(ratio * num_segments)
            splits.append(segments[start_idx:end_idx])
            start_idx = end_idx
        
        datasets = []
        for split in splits:
            indices = sum(split, [])
            subset = Subset(self, indices)
            datasets.append(subset)
        
        return datasets


def image_transform(path, torch_transforms):
    from PIL import Image
    image = Image.open(path).convert('RGB')
    image = np.array(image) / 255.0
    image = torch.tensor(image).permute(2, 0, 1)  # [C, H, W]
    image = torch_transforms(image)  # Normalize pixel values to [0, 1]
    return image

def safe_decoder(path):
    try:
        # first try frame-accurate seek
        return VideoDecoder(path, seek_mode="exact")
    except RuntimeError as e:
        # if seeking fails, retry with approximate mode
        return VideoDecoder(path, seek_mode="approximate")

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

def video_transform(path, time_start, time_end, torch_transforms, fps=30):
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
        frames = safe_get_frames(decoder, frame_indices)  # [num_frames, H, W, C]
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
        # Normalize and apply transforms
        frames = torch_transforms(frames)

    return frames
