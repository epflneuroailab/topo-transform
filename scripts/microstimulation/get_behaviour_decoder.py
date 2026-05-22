
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config
from data import AFRAZ2006
from data.neural_data import get_data_loader
from validate.neural_decoding import run_features, make_decoder
from validate.rois import nsd
from validate import load_transformed_model
from models import vit_transform

from utils import cached
from scripts.common import *
from .utils import _Dataset


class MultiLayerExtractor:
    def __call__(self, model, inputs, average_time=True):
        # Use the transformed topo model to get all target layers.
        layer_features, _ = model(inputs, do_transform=True)
        if average_time:
            layer_features = [feat.mean(dim=1) for feat in layer_features]  # B x C x H x W
        return torch.cat(layer_features, dim=1)  # B x (sum C) x H x W


def _get_decoder(ckpt_name, dataset_name):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    batch_size = 64

    if dataset_name == "afraz2006":
        dataset = AFRAZ2006(transforms=vit_transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    tr = _Dataset(dataset.trainset)
    train_loader = get_data_loader(tr, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    
    extractor = MultiLayerExtractor()
    train_features, train_labels = run_features(model, train_loader, extractor, device=device)
    decoder = make_decoder(test_type='classify', device=device, C=1e3)
    decoder.fit(train_features, train_labels)
    return decoder

def get_decoder(ckpt_name, dataset_name):
    return cached(f"get_decoder_multilayer_{dataset_name}_{ckpt_name}")(_get_decoder)(ckpt_name, dataset_name)


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    dataset_name = "afraz2006"
    decoder = get_decoder(ckpt_name, dataset_name)
