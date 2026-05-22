import torch
import numpy as np
from collections import defaultdict

from data.neural_data.collections.tang2025 import get_compilation
from data.neural_data import get_data_loader
from validate.neural_decoding import make_decoder, run_features
from validate.rois import nsd
from validate import load_transformed_model
from models import vit_transform

from utils import cached
from .common import LOCALIZER_P_THRESHOLD
from .common import LOCALIZER_T_THRESHOLD
from .common import MODEL_CKPT
from .get_localizers import get_localizer_human
from .get_localizers import get_localizer_model


class Extractor:
    def __init__(self):
        self.do_transform = True

    def activate_transform(self):
        self.do_transform = True

    def deactivate_transform(self):
        self.do_transform = False

    def __call__(self, model, inputs):
        with torch.no_grad():
            layer_features, layer_positions = model(inputs, do_transform=self.do_transform)
        return [lf.mean(dim=1) for lf in layer_features]  # average over time dimension


def _localizer_rsa(ckpt_name, rois, num_splits, fwhm_mm, resolution_mm, p_threshold=LOCALIZER_P_THRESHOLD, t_threshold=LOCALIZER_T_THRESHOLD):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, epoch = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    masks_model = get_localizer_model(rois, ckpt_name, p_thres=p_threshold, t_thres=t_threshold, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
    masks_human = get_localizer_human(rois)

    batch_size = 16
    ratios = (1., 0., 0.)
    mask = nsd.get_region_voxels(['high-ventral', 'high-lateral', 'high-dorsal'])
    print("Considering high-level cortex regions only. Voxel counts:", mask.sum())

    decoder = make_decoder(test_type='regress', device=device)

    scores = defaultdict(list)

    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        for split in range(num_splits):
            print(f"=== Split {split+1}/{num_splits} ===")
            seed = 42 + split   
            trainset, _, _ = get_compilation(vit_transform, ratios=ratios, seed=seed, type='clip')

            train_loader = get_data_loader(trainset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            extractor = Extractor()

            model_activities, neural_activities = run_features(
                model,
                train_loader,
                get_features=extractor,
                device=device,
            )

            model_activities = model_activities[0]

            # compute RSA between model and neural activations

            for i, roi, mask_model, mask_human in zip(range(len(rois)), rois, masks_model, masks_human):
                print(f"Processing ROI: {roi}")

                # Model activations for this ROI
                model_acts_roi = model_activities[:, mask_model[0]]  # shape: (n_samples, n_units)

                # Human neural activations for this ROI
                neural_acts_roi = neural_activities[:, 0, mask_human]  # shape: (n_samples, n_voxels)

                # remove nans
                nans = np.isnan(model_acts_roi).any(axis=0)
                model_acts_roi = model_acts_roi[:, ~nans]
                nans = np.isnan(neural_acts_roi).any(axis=0)
                neural_acts_roi = neural_acts_roi[:, ~nans]

                # Compute RDMs
                model_rdm = 1 - np.corrcoef(model_acts_roi)
                neural_rdm = 1 - np.corrcoef(neural_acts_roi)

                # Flatten upper triangles
                iu = np.triu_indices_from(model_rdm, k=1)
                model_vec = model_rdm[iu]
                neural_vec = neural_rdm[iu]

                # Compute correlation between RDMs
                rsa_score = np.corrcoef(model_vec, neural_vec)[0, 1]
                print(f"RSA score for ROI {roi}: {rsa_score:.4f}")

                scores[roi].append(rsa_score)
        
    scores = {roi: np.array(vals) for roi, vals in scores.items()}  # Convert lists to arrays
    return scores

def localizer_rsa(ckpt_name, rois, num_splits=1, fwhm_mm=0.0, resolution_mm=0.0):
    import hashlib
    rois_code = hashlib.md5('_'.join(sorted(rois)).encode()).hexdigest()[:8]
    return cached(f"localizer_rsa_splits{num_splits}_{ckpt_name}_rois{rois_code}_fwhm{fwhm_mm}_res{resolution_mm}")(_localizer_rsa)(ckpt_name, rois=rois, num_splits=num_splits, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

def main():
    ckpt_name = MODEL_CKPT
    num_splits = 1  # Adjust as needed

    rois = [
        'face',
        'v6',
        'psts',
        'mt',
    ]

    scores  = localizer_rsa(ckpt_name, rois, num_splits=num_splits)


if __name__ == "__main__":
    main()
