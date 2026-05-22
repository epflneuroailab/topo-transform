
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config
from data import AFRAZ2006
from data.neural_data import get_data_loader
from validate.rois import nsd
from validate import load_transformed_model
from models import vit_transform

from utils import cached
from scripts.common import *

from .utils import _Dataset
from .get_behaviour_decoder import get_decoder, MultiLayerExtractor
from .get_stimulation_location import *
from topo.perturbation import MicroStimulation, UnitAblation
from validate.neural_decoding import run_features, make_decoder


STIMULATION_PARAMETERS = {
    # "Microstimulation consisted of bipolar current pulses of 50mA delivered at 200 Hz (refs 19, 20).
    'current_pulse_mA': 5000,
    'pulse_rate_Hz': 2000,
}

ABLATION_PARAMETERS = {
    'ablation_radius_mm': 1.0,
}


def _build_cache_tag(stimulation_mode, perturbation_params):
    if perturbation_params is None:
        return stimulation_mode
    items = sorted(perturbation_params.items())
    params_tag = "_".join(f"{k}-{v}" for k, v in items)
    return f"{stimulation_mode}_{params_tag}"


def _test_stimulation(ckpt_name, dataset_name, stimulation_mode="amplify", perturbation_params=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    decoder = get_decoder(ckpt_name, dataset_name)

    if stimulation_mode == "amplify":
        stimulation = MicroStimulation(model)
        perturbation_params = perturbation_params or STIMULATION_PARAMETERS
    elif stimulation_mode == "ablate":
        stimulation = UnitAblation(model)
        perturbation_params = perturbation_params or ABLATION_PARAMETERS
    else:
        raise ValueError(f"Unknown stimulation_mode: {stimulation_mode}")

    batch_size = 64

    if dataset_name == "afraz2006":
        dataset = AFRAZ2006(transforms=vit_transform).valset
        roi = "face-afraz"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    shuffle = False
    assert shuffle == False, "Shuffle must be False for evaluation"
    val = _Dataset(dataset)
    val_loader = get_data_loader(val, batch_size=batch_size, shuffle=shuffle, num_workers=batch_size)
    label_signal_levels = dataset.label_signal_levels()
    
    extractor = MultiLayerExtractor()

    ret = {}

    # pre stimulation
    val_features, val_labels = run_features(model, val_loader, extractor, device=device)
    val_pred = decoder.predict_proba(torch.from_numpy(val_features))
    
    ret['pre_stimulation'] = val_pred[:,1]
    ret['label_signal_levels'] = label_signal_levels

    # post stimulation
    # stimulation_locations, selecitivities = get_selectivity_based_stimulation_locations(roi, ckpt_name, num_samples=50)
    stimulation_locations, sampled_indices = get_random_stimulation_locations(model, num_samples=100)

    val_pred_list = []
    for location in stimulation_locations:
        print(location)
        stimulation.perturb(location, perturbation_params)
        val_features, val_labels = run_features(model, val_loader, extractor, device=device)
        val_pred = decoder.predict_proba(torch.from_numpy(val_features))
        val_pred_list.append(val_pred[:,1])
        stimulation.clear()

    ret['post_stimulation'] = val_pred_list
    ret['stimulation_locations'] = stimulation_locations
    ret['sampled_indices'] = sampled_indices
    ret['stimulation_mode'] = stimulation_mode
    ret['perturbation_params'] = perturbation_params
    # ret['selecitivities'] = selecitivities

    return ret



def test_stimulation(ckpt_name, dataset_name, stimulation_mode="amplify", perturbation_params=None):
    cache_tag = _build_cache_tag(stimulation_mode, perturbation_params)
    return cached(f"test_stimulation_{dataset_name}_{ckpt_name}_{cache_tag}", rerun=True)(
        _test_stimulation
    )(ckpt_name, dataset_name, stimulation_mode=stimulation_mode, perturbation_params=perturbation_params)


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    dataset_name = "afraz2006"
    decoder = test_stimulation(ckpt_name, dataset_name, stimulation_mode="ablate")
