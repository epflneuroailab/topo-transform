import os
from pathlib import Path

from .utils import t_test


DEFAULT_ANIMACY_SIZE_DIR = "/mnt/upschrimpf2/scratch/ytang/datasets/konkle/AnimacySize"
FALLBACK_ANIMACY_SIZE_DIR = "/mnt/scratch/ytang/datasets/konkle/AnimacySize"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _resolve_animacy_size_dir(data_dir):
    if data_dir is None:
        data_dir = DEFAULT_ANIMACY_SIZE_DIR
    if os.path.exists(data_dir):
        return Path(data_dir)
    if data_dir == DEFAULT_ANIMACY_SIZE_DIR and os.path.exists(FALLBACK_ANIMACY_SIZE_DIR):
        return Path(FALLBACK_ANIMACY_SIZE_DIR)
    raise FileNotFoundError(f"AnimacySize dataset not found: {data_dir}")


def _collect_images(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Missing directory: {directory}")
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
    )


def konkle_animacy_size_datasets(data_dir=None):
    """Return datasets for Big/Small x Animate/Inanimate in AnimacySize."""
    root = _resolve_animacy_size_dir(data_dir)
    categories = {
        "big_animate": root / "Big-Animate",
        "big_inanimate": root / "Big-Inanimate",
        "small_animate": root / "Small-Animate",
        "small_inanimate": root / "Small-Inanimate",
    }

    datasets = {}
    for name, folder in categories.items():
        files = _collect_images(folder)
        if not files:
            raise ValueError(f"No images found for {name} in {folder}")
        datasets[name] = [(str(path), name) for path in files]

    return datasets


def konkle_size_datasets(data_dir=None):
    """Return datasets dict for small vs big using AnimacySize (pooled over animacy)."""
    datasets_4 = konkle_animacy_size_datasets(data_dir=data_dir)
    big_files = datasets_4["big_animate"] + datasets_4["big_inanimate"]
    small_files = datasets_4["small_animate"] + datasets_4["small_inanimate"]
    return {
        "big": [(path, "big") for path, _ in big_files],
        "small": [(path, "small") for path, _ in small_files],
    }


def konkle_animacy_datasets(data_dir=None):
    """Return datasets dict for animate vs inanimate using AnimacySize (pooled over size)."""
    datasets_4 = konkle_animacy_size_datasets(data_dir=data_dir)
    animate_files = datasets_4["big_animate"] + datasets_4["small_animate"]
    inanimate_files = datasets_4["big_inanimate"] + datasets_4["small_inanimate"]
    return {
        "animal": [(path, "animal") for path, _ in animate_files],
        "object": [(path, "object") for path, _ in inanimate_files],
    }


def localize_konkle_big_small(
    model,
    transform,
    data_dir=None,
    batch_size=32,
    device="cuda",
    downsampler=None,
    frames_per_video=24,
    video_fps=12,
    ret_pvals=False,
):
    datasets = konkle_size_datasets(data_dir=data_dir)
    t_vals_dict, p_vals_dict = t_test(
        model,
        transform,
        datasets,
        contrasts=[(1, -1)],
        batch_size=batch_size,
        device=device,
        downsampler=downsampler,
        frames_per_video=frames_per_video,
        video_fps=video_fps,
    )

    t_vals_ret = {"big_vs_small": t_vals_dict["big_vs_small"]}
    p_vals_ret = {"big_vs_small": p_vals_dict["big_vs_small"]}
    if ret_pvals:
        return t_vals_ret, p_vals_ret
    return t_vals_ret


def localize_konkle_animacy(
    model,
    transform,
    data_dir=None,
    batch_size=32,
    device="cuda",
    downsampler=None,
    frames_per_video=24,
    video_fps=12,
    ret_pvals=False,
):
    datasets = konkle_animacy_datasets(data_dir=data_dir)
    t_vals_dict, p_vals_dict = t_test(
        model,
        transform,
        datasets,
        contrasts=[(1, -1)],
        batch_size=batch_size,
        device=device,
        downsampler=downsampler,
        frames_per_video=frames_per_video,
        video_fps=video_fps,
    )

    t_vals_ret = {"animal_vs_object": t_vals_dict["animal_vs_object"]}
    p_vals_ret = {"animal_vs_object": p_vals_dict["animal_vs_object"]}
    if ret_pvals:
        return t_vals_ret, p_vals_ret
    return t_vals_ret


def localize_konkle(
    model,
    transform,
    data_dir=None,
    batch_size=32,
    device="cuda",
    downsampler=None,
    frames_per_video=24,
    video_fps=12,
    ret_pvals=False,
):
    big_small = localize_konkle_big_small(
        model,
        transform,
        data_dir=data_dir,
        batch_size=batch_size,
        device=device,
        downsampler=downsampler,
        frames_per_video=frames_per_video,
        video_fps=video_fps,
        ret_pvals=ret_pvals,
    )
    animacy = localize_konkle_animacy(
        model,
        transform,
        data_dir=data_dir,
        batch_size=batch_size,
        device=device,
        downsampler=downsampler,
        frames_per_video=frames_per_video,
        video_fps=video_fps,
        ret_pvals=ret_pvals,
    )

    if ret_pvals:
        t_big_small, p_big_small = big_small
        t_animacy, p_animacy = animacy
        t_vals = {}
        p_vals = {}
        t_vals.update(t_big_small)
        t_vals.update(t_animacy)
        p_vals.update(p_big_small)
        p_vals.update(p_animacy)
        return t_vals, p_vals

    t_vals = {}
    t_vals.update(big_small)
    t_vals.update(animacy)
    return t_vals
