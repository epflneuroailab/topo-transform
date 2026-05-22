from utils import cached
from models import vit_transform
from validate.smoothness import validate_smoothness
from validate import load_transformed_model


FLOC_DATASETS = ['vpnl', 'kanwisher', 'motion', 'pitzalis', 'biomotion', 'pitcher', 'temporal']

def _smoothness(
        checkpoint_name, 
        dataset_name, 
        device='cuda',
        frames_per_video=24,
        video_fps=12,
        fwhm_mm=2.0,
        resolution_mm=1.0,
    ):

    model, epoch = load_transformed_model(checkpoint_name=checkpoint_name, device=device)
    model.eval()
    transform = vit_transform
    ret = validate_smoothness(
        model,
        transform,
        dataset_name,
        device=device,
        frames_per_video=frames_per_video,
        video_fps=video_fps,
        fwhm_mm=fwhm_mm,
        resolution_mm=resolution_mm,
    )

    return ret


def smoothness(
        checkpoint_name, 
        dataset_name, 
        device='cuda',
        frames_per_video=24,
        video_fps=12,
        fwhm_mm=2.0,
        resolution_mm=1.0,
    ):
    cache_key = f"smoothness_{checkpoint_name}_{dataset_name}_{fwhm_mm}_{resolution_mm}"
    @cached(cache_key, rerun=False)
    def _cached_smoothness():
        return _smoothness(
            checkpoint_name, 
            dataset_name, 
            device=device,
            frames_per_video=frames_per_video,
            video_fps=video_fps,
            fwhm_mm=fwhm_mm,
            resolution_mm=resolution_mm,
        )
    return _cached_smoothness()