import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config
from data.neural_data.collections.tang2025 import get_compilation
from data.neural_data import Assembly, TemporalAssemblyDataset, get_data_loader
from validate.neural_decoding import decoding_test, make_decoder
from validate.rois import nsd
from validate import load_transformed_model
from topo import TopoTransformedVJEPA
from models import vit_transform

from utils import cached

class MaskedTargetLoader(Dataset):
    def __init__(self, data_loader, target_mask):
        self.data_loader = data_loader
        self.target_mask = target_mask

    def __len__(self):
        return len(self.data_loader)

    def __getitem__(self, idx):
        data, target = self.data_loader[idx]
        masked_target = target[..., self.target_mask]
        return data, masked_target

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

def _neural_alignment(ckpt_name, num_splits):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    batch_size = 16
    ratios = (0.8, 0.1, 0.1)
    mask = nsd.get_region_voxels(['high-ventral', 'high-lateral', 'high-dorsal'])
    print("Considering high-level cortex regions only. Voxel counts:", mask.sum())

    decoder = make_decoder(test_type='regress', device=device)

    scores_pretransform = []
    scores_posttransform = []

    for split in range(num_splits):
        print(f"=== Split {split+1}/{num_splits} ===")
        seed = 42 + split   
        trainset, valset, testset, ceiling = get_compilation(vit_transform, ratios=ratios, return_ceiling=True, seed=seed)

        ceil = ceiling[..., mask].mean()
        print(f"Ceiling: {ceil}")

        tr = MaskedTargetLoader(trainset, mask)
        va = MaskedTargetLoader(valset, mask)
        te = MaskedTargetLoader(testset, mask)

        train_loader = get_data_loader(tr, batch_size=batch_size, shuffle=True, num_workers=batch_size)
        val_loader = get_data_loader(va, batch_size=batch_size, shuffle=False, num_workers=batch_size)
        test_loader = get_data_loader(te, batch_size=batch_size, shuffle=False, num_workers=batch_size)
        
        extractor = Extractor()

        for mode in ['pretransform', 'posttransform']:
            if mode == 'pretransform':
                extractor.deactivate_transform()
            else:
                extractor.activate_transform()
            print(f"Evaluating mode: {mode}")

            scores = decoding_test(
                model=model,
                get_features=extractor,
                train_loaders=[train_loader],
                test_loaders=[val_loader, test_loader],
                downsampler=None,
                decoder=decoder,
                device=device,
            )

            validated_score, layer_idx = _val_selection(scores)

            print(f"Score: {validated_score.mean()}")
            for i in range(model.num_layers):
                perc = (np.array(layer_idx) == i).mean() * 100
                print(f"  Layer {i}: selected {perc:.1f}%")

            if mode == 'pretransform':
                scores_pretransform.append(validated_score)
            else:
                scores_posttransform.append(validated_score)

    scores_pretransform = np.array(scores_pretransform)  # [num_splits, num_voxels]
    scores_posttransform = np.array(scores_posttransform)  # [num_splits, num_voxels]
    return scores_pretransform, scores_posttransform, mask, ceiling

def neural_alignment(ckpt_name, num_splits=1):
    return cached(f"neural_alignment_splits{num_splits}_{ckpt_name}")(_neural_alignment)(ckpt_name, num_splits=num_splits)