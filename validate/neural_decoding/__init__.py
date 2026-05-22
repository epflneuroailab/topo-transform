import copy

import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from .cross_validation import make_cv
from .decoder import make_decoder
from .downsampler import make_downsampler

SUBMODULE_SEPARATOR = '.'

"""
This module implements the decoding-test for any model.

An offline decoding test process should specify the following:
1. decoder type: e.g. linear regression, ridge regression, logistic regression, etc. A decoder implements `fit`, `predict`, and `score` methods.
2. layer choices: the decoder will choose the layer with the highest validation accuracy for decoding; for every decoding target (e.g., #regression outputs).
3. cross-validation settings: e.g. number of folds, split percentages, stratified or not, etc.
4. downsampling: for offline decoding, the dataset may be too large to fit in memory; we may need to downsample the dataset.

get_features(model, inputs) -> features [B, num_features] or list([B, num_features])
"""


def decoding_test(
    model,
    get_features,
    train_loaders,
    test_loaders,
    decoder,
    downsampler=None,
    device=None,
    ddp_enabled=False,
    rank=0,
    ret_decoders=False,  # cost GPU memory
    debug=False,
    debug_batches=2,
):
    """Train decoders on multiple train loaders and evaluate on multiple test loaders.
    
    If a loader appears in both train and test, features are only computed once.
    
    Returns:
        scores: np.ndarray of shape [num_sources, num_trains, num_tests, *units/time]
        decoders: nested list of trained decoders [num_sources][num_trains]
    """

    # Normalize to lists
    if not isinstance(train_loaders, (list, tuple)):
        train_loaders = [train_loaders]
    if not isinstance(test_loaders, (list, tuple)):
        test_loaders = [test_loaders]

    def _run_features(loader):
        return run_features(
            model, loader, get_features,
            device, downsampler=downsampler,
            rank=rank, ddp_enabled=ddp_enabled,
            debug=debug, debug_batches=debug_batches,
        )

    # Collect all features
    train_feats, train_targets = [], []
    for loader in train_loaders:
        feat, target = _run_features(loader)
        train_feats.append(feat)
        train_targets.append(target)

    test_feats, test_targets = [], []
    for loader in test_loaders:
        feat, target = _run_features(loader)
        test_feats.append(feat)
        test_targets.append(target)

    # Handle single-feature case
    if not isinstance(train_feats[0], list):
        train_feats = [[f] for f in train_feats]     # [num_trains][num_sources][B, F]
        test_feats = [[f] for f in test_feats]       # [num_tests][num_sources][B, F]

    # Decode
    tmp = decode(
        train_feats, train_targets,
        test_feats, test_targets,
        decoder,
        ret_decoders=ret_decoders,
    )

    if ret_decoders:
        scores, decoders = tmp
        return scores, decoders
    else:
        scores = tmp
        return scores


def decode(train_feats, train_targets, test_feats, test_targets, decoder, ret_decoders=False):
    """Decode from multiple feature sources.
    Args:
        train_feats: list of list of np.ndarray, [num_trains][num_sources][num_samples, num_features]
        train_targets: list of np.ndarray, [num_trains][num_samples, *units/time]
        test_feats: list of list of np.ndarray, [num_tests][num_sources][num_samples, num_features]
        test_targets: list of np.ndarray, [num_tests][num_samples, *units/time]
        decoder: decoder object implementing fit and score methods
        ret_decoders: whether to return trained decoders
    Returns:
        scores: np.ndarray of shape [num_sources, num_trains, num_tests, *units/time]
        decoders_out: nested list of trained decoders [num_sources][num_trains]
    """

    num_sources = len(train_feats[0])
    num_trains = len(train_feats)
    num_tests = len(test_feats)

    scores = None
    decoders_out = [[None for _ in range(num_trains)] for _ in range(num_sources)]

    for s in range(num_sources):
        for t in range(num_trains):
            x_train = train_feats[t][s]
            y_train = train_targets[t]

            dec = copy.deepcopy(decoder)
            dec.fit(x_train, y_train)
            decoders_out[s][t] = dec if ret_decoders else None

            for u in range(num_tests):
                x_test = test_feats[u][s]
                y_test = test_targets[u]
                result = dec.score(x_test, y_test)

                # initialize scores with correct shape on first pass
                if scores is None:
                    scores = np.zeros(
                        (num_sources, num_trains, num_tests) + np.shape(result),
                        dtype=float,
                    )
                scores[s, t, u] = result

    if ret_decoders:
        return scores, decoders_out
    else:
        return scores


def run_features(model, data_loader, get_features, device, downsampler=None, rank=0, ddp_enabled=False, debug=False, debug_batches=2):
    """Run feature extraction on a data loader.
    Returns:
        offline_feats: np.ndarray of shape [num_samples, num_features] or list of np.ndarray
        offline_targets: np.ndarray of shape [num_samples, *units/time]
    """
    def get_features_wrapper(model, x):
        outputs = get_features(model, x)
        if downsampler is not None:
            if isinstance(outputs, (list, tuple)):
                outputs = [downsampler(out) for out in outputs]
            else:
                outputs = downsampler(outputs)
        return outputs

    if device is None: device = model.device
    model.eval()
    local_feats = []
    local_targets = []

    for i, (data, target) in enumerate(tqdm(data_loader)):
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = get_features_wrapper(model, data)
            ret_list = isinstance(outputs, (list, tuple))

        if ret_list:
            outputs = [out.cpu().numpy() for out in outputs]
            if len(local_feats) == 0:
                local_feats = [[] for _ in range(len(outputs))]
            for j, out in enumerate(outputs):
                local_feats[j].append(out)
        else:
            local_feats.append(outputs.cpu().numpy())
        local_targets.append(target.cpu().numpy())

        if debug:
            if i >= debug_batches:
                break

    # --- Synchronize and gather features across all ranks ---
    if ddp_enabled:
        dist.barrier()

        world_size = dist.get_world_size()
        all_feats = [None for _ in range(world_size)]
        all_targets = [None for _ in range(world_size)]
        all_indices = [None for _ in range(world_size)]

        dist.all_gather_object(all_feats, local_feats)
        dist.all_gather_object(all_targets, local_targets)

        offline_feats = []
        offline_targets = []

        if ret_list:
            offline_feats = [[] for _ in range(len(all_feats[0]))]
            for rank_feats in all_feats:
                for j, rank_feat in enumerate(rank_feats):
                    offline_feats[j].extend(rank_feat)
        else:
            for rank_feats in all_feats:
                offline_feats.extend(rank_feats)

        for rank_targets in all_targets:
            offline_targets.extend(rank_targets)

    else:
        offline_feats = local_feats
        offline_targets = local_targets

    # --- Only Rank 0 does decoding & reporting ---
    if rank != 0:
        return

    if ret_list:
        offline_feats = [np.concatenate(of, 0) for of in offline_feats]
    else:
        offline_feats = np.concatenate(offline_feats, 0)
    offline_targets = np.concatenate(offline_targets, 0)

    return offline_feats, offline_targets


__all__ = [
    "SUBMODULE_SEPARATOR",
    "decode",
    "decoding_test",
    "make_cv",
    "make_decoder",
    "make_downsampler",
    "run_features",
]
