import torch
import numpy as np
from data.neural_data.collections.tang2025 import get_compilation
from data.neural_data import get_data_loader
from validate.neural_decoding import decode, make_decoder, run_features
from validate.rois import nsd
from validate import load_transformed_model
from models import vit_transform

from utils import cached
from .analysis_utils import collect_by_ckpt
from .common import LOCALIZER_P_THRESHOLD
from .common import LOCALIZER_T_THRESHOLD
from .common import SWAPOPT_CKPTS
from .common import UNOPTIMIZED_CKPTS
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


def _localizer_decode(ckpt_name, rois, num_splits, fwhm_mm, resolution_mm, p_threshold=LOCALIZER_P_THRESHOLD, t_threshold=LOCALIZER_T_THRESHOLD):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, epoch = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    masks_model = get_localizer_model(rois, ckpt_name, p_thres=p_threshold, t_thres=t_threshold, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
    masks_human = get_localizer_human(rois)

    batch_size = 16
    ratios = (.9, .1, 0.)
    mask = nsd.get_region_voxels(['high-ventral', 'high-lateral', 'high-dorsal'])
    print("Considering high-level cortex regions only. Voxel counts:", mask.sum())

    decoder = make_decoder(test_type='regress', device=device)

    all_scores = []
    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        for split in range(num_splits):
            print(f"=== Split {split+1}/{num_splits} ===")
            seed = 42 + split   
            trainset, testset, _ = get_compilation(vit_transform, ratios=ratios, seed=seed, type='clip')

            train_loader = get_data_loader(trainset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            test_loader = get_data_loader(testset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            extractor = Extractor()

            train_model_activities, train_neural_activities = run_features(
                model,
                train_loader,
                get_features=extractor,
                device=device,
            )

            test_model_activities, test_neural_activities = run_features(
                model,
                test_loader,
                get_features=extractor,
                device=device,
            )

            train_model_activities = train_model_activities[0]
            test_model_activities = test_model_activities[0]

            # compute RSA between model and neural activations

            train_model_acts = []
            train_neural_acts = []
            test_model_acts = []
            test_neural_acts = []

            for i, roi, mask_model, mask_human in zip(range(len(rois)), rois, masks_model, masks_human):
                print(f"Processing ROI: {roi}")

                # Model activations for this ROI
                train_model_acts_roi = train_model_activities[:, mask_model[0]]  # shape: (n_samples, n_units)
                test_model_acts_roi = test_model_activities[:, mask_model[0]]  # shape: (n_samples, n_units)

                # Human neural activations for this ROI
                train_neural_acts_roi = train_neural_activities[:, 0, mask_human]  # shape: (n_samples, n_voxels)
                test_neural_acts_roi = test_neural_activities[:, 0, mask_human]  # shape: (n_samples, n_voxels)

                train_model_acts.append([train_model_acts_roi])
                train_neural_acts.append(train_neural_acts_roi)
                test_model_acts.append([test_model_acts_roi])
                test_neural_acts.append(test_neural_acts_roi)
                
            decoder = make_decoder('regress', device)

            scores = np.zeros((len(rois), len(rois)))
            for i, roi_a in enumerate(rois):
                for j, roi_b in enumerate(rois):
                    print(f"Decoding ROI: model {roi_a} to human {roi_b}")

                    train_model_roi = [train_model_acts[i]]
                    train_neural_roi = [train_neural_acts[j]]
                    test_model_roi = [test_model_acts[i]]
                    test_neural_roi = [test_neural_acts[j]]

                    decode_scores = decode(
                        train_model_roi,
                        train_neural_roi,
                        test_model_roi,
                        test_neural_roi,
                        decoder
                    )

                    scores[i, j] = decode_scores.mean().item()  # average over all voxels

            all_scores.append(scores)

    all_scores = np.array(all_scores)  # shape: (num_splits, num_rois, num_rois)

    return all_scores

def localizer_decode(ckpt_name, rois, num_splits=1, fwhm_mm=2.0, resolution_mm=1.0):
    import hashlib
    rois_code = hashlib.md5('_'.join(sorted(rois)).encode()).hexdigest()[:8]
    return cached(f"localizer_decode_splits{num_splits}_{ckpt_name}_rois{rois_code}_fwhm{fwhm_mm}_res{resolution_mm}", rerun=False)(_localizer_decode)(ckpt_name, rois=rois, num_splits=num_splits, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

def main():
    num_splits = 1  # Adjust as needed

    rois = [
        'face',
        'place',
        'body',
        'v6',
        'psts',
        'mt',
    ]

    for ckpt_list in (UNOPTIMIZED_CKPTS, SWAPOPT_CKPTS):
        collect_by_ckpt(ckpt_list, localizer_decode, rois, num_splits=num_splits)


if __name__ == "__main__":
    main()
