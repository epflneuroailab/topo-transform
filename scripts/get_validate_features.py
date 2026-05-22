from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from data import Kinetics400
from topo import TopoTransformedVJEPA, TopoTransformedTDANN

from utils import cached
from validate import load_transformed_model
from models import vit_transform

def _validate_features(ckpt_name):

    is_swapopt = ckpt_name == "swapopt"

    # Load data
    data = Kinetics400(train_transforms=vit_transform, test_transforms=vit_transform, fps=12)
    val_loader = DataLoader(data.valset, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_transformed_model(ckpt_name, device=device)[0]
    model.eval()

    # Extract features
    all_features = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Extracting features'):
            videos = batch[0].to(device)
            features, positions = model(videos)
            all_features.append([f.cpu() for f in features])
    all_features = [torch.cat([batch[l] for batch in all_features], dim=0) for l in range(len(all_features[0]))]
    return all_features, positions

def validate_features(ckpt_name: str):
    return cached('validate_features', rerun=False)(_validate_features)(ckpt_name)