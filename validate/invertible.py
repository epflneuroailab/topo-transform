import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import copy

from utils import cached
from data.smthsmthv2 import SmthSmthV2

def validate_invertibility(model, transform, dataset_name='ssv2', duration=2000, fps=12,
                          batch_size=32, device='cuda', num_samples=4, seed=42):

    # Get layer positions for visualization
    layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]
    
    # Get dataset
    if dataset_name == 'ssv2':
        smthsmth = SmthSmthV2(train_transforms=transform, test_transforms=transform, 
                             duration=duration, fps=fps)
        dataset = smthsmth.valset
        dataset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
    
    print(f"Processing {len(dataset)} videos to test invertibility (seed={seed})")
    
    # Create dataloaders
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=int(batch_size/1.5),
        shuffle=False,
        pin_memory=True
    )

    model.eval()
    for tmp in tqdm(loader, desc="Checking invertibility"):
        data, labels = tmp[0], tmp[1]
        data = data.to(device)

        with torch.no_grad():
            layer_features = model.extractor.extract_features(model.model, data)
            transformed_features = model.transform(layer_features)
            inverted_features = model.transform.inverse(transformed_features)

            abs_errs = []
            for layer_feat, inverted_feat in zip(layer_features, inverted_features):
                abs_err = torch.abs(layer_feat - inverted_feat).mean().item()
                abs_errs.append(abs_err)
            
            mean_abs_errs = torch.tensor(abs_errs).mean().item()
            print(f"Mean absolute error across layers: {mean_abs_errs:.6f}")
            assert mean_abs_errs < 1e-4, f"Invertibility test failed: mean abs error {mean_abs_errs} too high"