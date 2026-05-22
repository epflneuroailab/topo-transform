import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config
from data.neural_data import Assembly, TemporalAssemblyDataset, get_data_loader
from validate.neural_decoding import decoding_test, make_decoder
from validate.rois import nsd
from validate import load_transformed_model
from models import vit_transform

from data import SmthSmthV2, ImageNetVid

from utils import cached


class _Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label, _ = self.dataset[idx]
        return data, label

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

def _task_performance(ckpt_name, dataset_name):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    batch_size = 16
    decoder = make_decoder(test_type='classify', device=device)

    scores_pretransform = []
    scores_posttransform = []

    if dataset_name == "ssv2":
        dataset = SmthSmthV2(subsample_factor=0.1, train_transforms=vit_transform, test_transforms=vit_transform)
    elif dataset_name == "imagenet":
        dataset = ImageNetVid(subsample_factor=0.05, train_transforms=vit_transform, test_transforms=vit_transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    tr = _Dataset(dataset.trainset)
    va = _Dataset(dataset.valset)

    train_loader = get_data_loader(tr, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    val_loader = get_data_loader(va, batch_size=batch_size, shuffle=False, num_workers=batch_size)
    
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
            test_loaders=[val_loader],
            downsampler=None,
            decoder=decoder,
            device=device,
        )

        validated_score = scores.item()
        print(f"Mode {mode}: {validated_score}")

        if mode == 'pretransform':
            scores_pretransform = validated_score
        else:
            scores_posttransform = validated_score

    return scores_pretransform, scores_posttransform

def task_performance(ckpt_name, dataset_name):
    return cached(f"task_performance_{dataset_name}_{ckpt_name}")(_task_performance)(ckpt_name, dataset_name)