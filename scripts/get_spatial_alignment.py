import torch
import numpy as np
from torch.utils.data import Dataset

from data.neural_data.collections.tang2025 import get_compilation
from data.neural_data import get_data_loader
from validate.neural_decoding import make_decoder, run_features
from validate.rois import nsd
from validate import load_transformed_model
from models import vit_transform

from utils import cached
from .common import MODEL_CKPT

from validate.smoothness import compute_activity_smoothness_model, compute_activity_smoothness_neural

class TargetLoader(Dataset):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, idx):
        data, target = self.data_loader[idx]
        target = compute_activity_smoothness_neural(target)
        return data, target

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

def _val_selection(decoding_scores):
    val_scores, test_scores = decoding_scores[:,0,0], decoding_scores[:,0,1]
    best_val_idx = val_scores.argmax(0)
    test_score = test_scores[best_val_idx, np.arange(test_scores.shape[1])]
    return test_score, best_val_idx

def _spatial_alignment(ckpt_name, num_splits, fwhm_mm, resolution_mm):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, epoch = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    batch_size = 16
    ratios = (1, 0., 0.)
    mask = nsd.get_region_voxels(['high-ventral', 'high-lateral', 'high-dorsal'])
    print("Considering high-level cortex regions only. Voxel counts:", mask.sum())

    decoder = make_decoder(test_type='regress', device=device)

    scores = []

    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        for split in range(num_splits):
            print(f"=== Split {split+1}/{num_splits} ===")
            seed = 42 + split   
            trainset, _, _ = get_compilation(vit_transform, ratios=ratios, seed=seed, type='clip')

            tr = TargetLoader(trainset)
            train_loader = get_data_loader(tr, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            extractor = Extractor()

            model_activities, neural_moran = run_features(
                model,
                train_loader,
                get_features=extractor,
                device=device,
            )

            model_moran = compute_activity_smoothness_model(model_activities[0][:,0], model)

            # remove nan
            neural_moran = neural_moran.flatten()
            model_moran = model_moran.flatten()
            valid_sel = ~np.isnan(neural_moran) & ~np.isnan(model_moran)
            model_moran = model_moran[valid_sel]
            neural_moran = neural_moran[valid_sel]

            corr = np.corrcoef(model_moran, neural_moran)[0,1]
            print(f"Split {split+1} correlation between model and neural smoothness: {corr:.4f}")
            scores.append(corr)

    scores = np.array(scores)  # [num_splits, num_voxels]
    return scores

def spatial_alignment(ckpt_name, num_splits=1, fwhm_mm=2.0, resolution_mm=1.0):
    return cached(f"spatial_alignment_splits{num_splits}_{ckpt_name}_fwhm{fwhm_mm}_res{resolution_mm}")(_spatial_alignment)(ckpt_name, num_splits=num_splits, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

def main():
    ckpt_name = MODEL_CKPT
    num_splits = 1  # Adjust as needed
    scores  = spatial_alignment(ckpt_name, num_splits=num_splits)


if __name__ == "__main__":
    main()
