import os
import av
import cv2
import numpy as np
# from torchcodec.decoders import VideoDecoder
from PIL import Image as PILImage
from typing import Union, Tuple, List, Dict, Any, Optional, Sequence, Type, Callable
from pathlib import Path
import hashlib

EPS = 1e-9
CACHE = {}
CACHE_DIR = "/mnt/scratch/ytang/datasets/TMP"


# cv2 has the wierd bug of cannot handling too large channel size
def cv2_resize(arr, size, mode, batch_size=4):
    # arr [H, W, C]
    import cv2
    ori_dtype = arr.dtype
    arr = arr.astype(float)
    C = arr.shape[-1]
    ret = []
    for i in range(0, C, batch_size):
        val = cv2.resize(arr[..., i:i+batch_size], size, interpolation=mode)
        if len(val.shape)<3: val = val[...,None]
        ret.append(val)
    return np.concatenate(ret, axis=-1).astype(ori_dtype)

def batch_2d_resize(arr, size, mode):
    # arr [N, H, W, C]
    N, H, W, C = arr.shape
    arr = arr.transpose(1, 2, 3, 0).reshape(H, W, C*N)
    if mode == "bilinear":
        mode = cv2.INTER_LINEAR
        ret = cv2_resize(arr, size, mode)
    elif mode == "pool":
        # ret = proportional_average_pooling(arr, size)
        mode = cv2.INTER_AREA
        ret = cv2_resize(arr, size, mode)
    else:
        raise ValueError(f"Unknown mode {mode}")
    ret = ret.reshape(size[1], size[0], C, N).transpose(3, 0, 1, 2)
    return ret

# def get_video_stats(video_path):
#     assert os.path.exists(video_path), f"Video file {video_path} does not exist."
#     cap = cv2.VideoCapture(video_path)
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = cap.get(cv2.CAP_PROP_FPS)
#     size = (width, height)
#     duration = length / fps * 1000 if fps > 0 else 0
#     cap.release()
#     return fps, duration, size

## NOTE: av video stats are much more accurate than opencv
def get_video_stats(video_path):
    if video_path in CACHE:
        return CACHE[video_path]

    assert os.path.exists(video_path), f"Video file {video_path} does not exist."

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Get FPS
    if stream.average_rate is not None:
        fps = float(stream.average_rate)
    else:
        # Fallback: estimate from time_base and duration
        fps = 1.0 / stream.time_base if stream.time_base else 0

    # Get number of frames
    frame_count = stream.frames
    if frame_count == 0:
        # fallback by decoding
        frame_count = sum(1 for _ in container.decode(stream))

    # Get frame size
    width = stream.codec_context.width
    height = stream.codec_context.height
    size = (width, height)

    # Get duration in milliseconds
    if stream.duration is not None:
        duration = stream.duration * stream.time_base * 1000  # in ms
    else:
        duration = frame_count / fps * 1000 if fps > 0 else 0

    # Cache the result
    CACHE[video_path] = (fps, duration, size)

    return fps, duration, size


def get_image_stats(image_path):
    with PILImage.open(image_path) as img:
        return img.size
    return size

def get_image_size(path):
    with PILImage.open(path) as img:
        size = img.size
    return size

class Image:
    def __init__(self, path: str, size: int):
        self._path = path
        self._size = size

    def copy(self):
        return Image(self._path, self._size)
    
    @property
    def size(self):
        return self._size
    
    def set_size(self, size):
        img = self.copy()
        img._size = size
        return img
    
    def from_path(path):
        return Image(path, get_image_size(path))
    
    def to_pil_img(self):
        return PILImage.fromarray(self.to_numpy())

    def get_frame(self):
        return np.array(PILImage.open(self._path).convert('RGB'))
    
    # return (H, W, C[RGB])
    def to_numpy(self):
        arr = self.get_frame()

        if arr.shape[:2][::-1] != self._size:
            arr = batch_2d_resize(arr[None,:], self._size, "bilinear")[0]

        return arr

    def store_to_path(self, path):
        self.to_img().save(path)
        return path

class Video:
    """Video object that represents a video clip."""

    def __init__(
            self, 
            path: Union[str, Path], 
            fps: float, 
            start: float, 
            end: float, 
            size: Tuple[int, int]
        ):
        self._path = path
        self._fps = fps
        self._size = size
        self._start = start
        self._end = end
        self._original_fps = None
        self._original_duration = None
        self._original_size = None

    def __getattribute__(self, key):
        if key.startswith("_original_"):
            if super().__getattribute__(key) is None:
                self._original_fps, self._original_duration, self._original_size = get_video_stats(self._path)
        return super().__getattribute__(key)

    def copy(self):
        # return view
        video = self.__class__(self._path, self._fps, self._start, self._end, self._size)
        video._original_fps = self._original_fps
        video._original_duration = self._original_duration
        video._original_size = self._original_size
        return video

    def __repr__(self):
        # unique string that encode all information
        st = f"{hashlib.sha256(self._path.encode()).hexdigest()}_{self._fps:.3f}fps_{self._start:.3f}ms_{self._end:.3f}ms_{self._size[0]}x{self._size[1]}"
        return st
    
    @property
    def duration(self):
        # in ms
        return self._end - self._start

    @property
    def fps(self):
        return self._fps

    @property
    def num_frames(self):
        return int(self.duration * self.fps/1000 + EPS)
    
    @property
    def original_num_frames(self):
        return int(self._original_duration * self._original_fps/1000 + EPS)
    
    @property
    def frame_size(self):
        return self._size
    
    ### Transformations: return copy
    
    def set_fps(self, fps):
        assert 1000/fps <= self.duration, f"fps {fps} is too low for duration {self.duration}ms."
        video = self.copy()
        video._fps = fps
        return video
    
    def set_size(self, size):
        # size: (width, height)
        video = self.copy()
        video._size = size
        return video

    def set_window(self, start, end, padding="repeat"):
        # use ms as the time scale
        if end < start:
            raise ValueError("end time is earlier than start time")
        
        if padding != "repeat":
            raise NotImplementedError()
        
        video = self.copy()
        video._start = self._start + start
        video._end = self._start + end
        return video

    def _check_indices_ascending(self, indices):
        if len(indices) == 0:
            return False
        if len(indices) == 1:
            return True
        for i in range(1, len(indices)):
            if indices[i] < indices[i-1]:
                return False
        return True

    def _sanitize_frames(self, frames, tol=0.9):
        # check if the read frames are valid
        # if some last frames are invalid, just copy the last valid frame
        # default tolerance: 0.01 of the total duration
        num_invalid = sum([f is None for f in frames])
        if num_invalid == len(frames): raise ValueError("No valid frames.")
        for i in range(num_invalid): assert frames[-1-i] is None, "Invalid frames are not at the end."
        if num_invalid > int(self._original_duration / 1000 * self._fps * tol): raise ValueError("Too many invalid frames.")
        if num_invalid > 0: 
            for i in range(num_invalid):
                frames[-1-i] = frames[-1-num_invalid]
            print(f"Warning: last {num_invalid} frames are invalid.")
        return frames

    def get_frames(self, indices):
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self._path}")

        def _read(cap):
            ret, frame = cap.read()
            if not ret:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ascending read optimization
        frames = []
        if self._check_indices_ascending(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])  # Move to the first frame index
            frame_index = indices[0] - 1
            for target_index in indices:
                to_move = target_index - frame_index
                for _ in range(to_move): frame = _read(cap)
                frames.append(frame)
                frame_index += to_move
        else:
            # random access
            for i, index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # Move to the frame index
                frames.append(_read(cap)) 

        cap.release()
        frames = self._sanitize_frames(frames)

        return np.array(frames)

    # def get_frames(self, indices):
    #     """
    #     indices: list of frame indices to extract (0-based).
    #     """
    #     if len(indices) == 0:
    #         return np.array([])

    #     video = VideoDecoder(self._path)
    #     frames = video.get_frames_at(indices).data.permute(0, 2, 3, 1).numpy()

    #     return frames

    def to_tensor(self, cache=False):
        # try:
        #     if self._path.endswith(".avi"): raise ValueError("avi format is not supported.")
        #     return self._to_tensor()
        # except Exception as e:
        #     print(f"Error in to_tensor: {e}")


        # check if saved
        target_path = os.path.join(CACHE_DIR, f"{self.__repr__()}.npy")

        import torch
        if cache:
            if os.path.exists(target_path):
                # Fast load from cache
                data = np.load(target_path, allow_pickle=False)
                return torch.from_numpy(data)
            else:
                # Save to cache atomically
                data = self.to_numpy()
                np.save(target_path, data)
                return torch.from_numpy(data)
        else:
            return torch.from_numpy(self.to_numpy())

    def _to_tensor(self):
        # get the time stamps of frame samples
        start_frame = self._start * self._original_fps / 1000
        end_frame = self._end * self._original_fps / 1000
        # avoid taking the last extra frame
        samples = np.arange(start_frame, end_frame - EPS, self._original_fps/self.fps)
        sample_indices = samples.astype(int)

        # padding: repeat the first/last frame
        original_num_frames = int(self._original_duration * self._original_fps/1000 - EPS)  # EPS to avoid last frame OOB error
        sample_indices = np.clip(sample_indices, 0, original_num_frames-1)

        # actual sampling
        video = VideoDecoder(self._path)
        frames = video.get_frames_at(sample_indices).data

        import torch.nn.functional as F
        if self._size != (frames.shape[2], frames.shape[1]):
            frames = F.interpolate(frames, size=(self._size[1], self._size[0]), mode='bilinear', align_corners=False)

        return frames.permute(0, 2, 3, 1)  # [batch, H, W, C]

    ### I/O
    def from_path(path):
        fps, end, size = get_video_stats(path)
        start = 0
        return Video(path, fps, start, end, size)
    
    def from_img_path(img_path, duration, fps):
        # duration in ms
        size = get_image_stats(img_path)
        return VideoFromImage(img_path, fps, 0, duration, size)

    def from_img_paths(img_paths: List[str], fps: float):
        # img_paths: list of image paths
        if len(img_paths) == 0:
            raise ValueError("Image paths cannot be empty.")
        return VideoFromImages(img_paths, fps)
    
    def to_numpy(self):
        # get the time stamps of frame samples
        start_frame = self._start * self._original_fps / 1000
        end_frame = self._end * self._original_fps / 1000
        # avoid taking the last extra frame
        samples = np.arange(start_frame, end_frame - EPS, self._original_fps/self.fps)
        sample_indices = samples.astype(int)

        # padding: repeat the first/last frame
        original_num_frames = int(self._original_duration * self._original_fps/1000 - EPS)  # EPS to avoid last frame OOB error
        sample_indices = np.clip(sample_indices, 0, original_num_frames-1)

        # actual sampling
        frames = self.get_frames(sample_indices)

        # resizing
        if self._size != (frames.shape[2], frames.shape[1]):
            frames = batch_2d_resize(frames, self._size, "bilinear")

        return frames
    
    def to_frames(self):
        return [f for f in self.to_numpy()]

    def to_pil_imgs(self):
        return [PILImage.fromarray(frame) for frame in self.to_numpy()]

    def to_path(self):
        # use context manager ?
        path = None  # make a temporal file
        raise NotImplementedError()
        return path

    def store_to_path(self, path):
        # pick format based on path filename
        if path.endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif path.endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise ValueError("Unsupported video format.")

        out = cv2.VideoWriter(path, fourcc, self._fps, self._size)
        for frame in self.to_frames():
            out.write(frame[...,::-1])  # to RGB
        out.release()
        return path
    
class VideoFromImage(Video):
    def get_frames(self, indices):
        data = Image.from_path(self._path).to_numpy()
        N = len(indices)
        ret = np.repeat(data[np.newaxis, ...], N, axis=0)
        return ret

class VideoFromImages(Video):
    def __init__(self, paths: List[str], fps: float):
        self._paths = paths
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Image path {path} does not exist.")
        
        num_frames = len(paths)
        start = 0
        end = num_frames / fps * 1000
        size = get_image_size(paths[0])
        super().__init__("", fps, start, end, size)
        self._original_fps = fps
        self._original_duration = end
        self._original_size = size

    def copy(self):
        video = VideoFromImages(self._paths, self._original_fps)
        video._start = self._start
        video._end = self._end
        video._size = self._size
        video._fps = self._fps
        video._original_fps = self._original_fps
        video._original_duration = self._original_duration
        video._original_size = self._original_size
        return video

    def get_frames(self, indices):
        frames = []
        for idx in indices:
            if idx < 0 or idx >= len(self._paths):
                frames.append(None)
            else:
                img = Image.from_path(self._paths[idx])
                frames.append(img.to_numpy())
        frames = self._sanitize_frames(frames)
        return np.array(frames)


if __name__ == "__main__":
    i=140
    # for i in range(1, 1000):
    print(i)
    video = Video.from_path(f"/mnt/scratch/fkolly/datasets/smthsmthv2/videos/{i}.webm").set_fps(12).set_window(1000, 4000)
    print((video.to_numpy().astype(float)-video.to_tensor().numpy().astype(float)))
    video = Video.from_path('/mnt/scratch/fkolly/datasets/AFD101/videos/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi').set_fps(12).set_window(1000, 4000)
    print((video.to_numpy().astype(float)-video.to_tensor().numpy().astype(float)))
    print()
    breakpoint()