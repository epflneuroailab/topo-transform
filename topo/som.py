import math

import numpy as np
import torch

from spacetorch.models.positions import LayerPositions


def som_name(layer_indices, unit_mm=1.0):
    layer_id = "_".join(str(layer_index) for layer_index in layer_indices)
    unit_str = f"{unit_mm:g}".replace(".", "p")
    return f"som_vjepa_{layer_id}_single_{unit_str}mm"


def som_checkpoint_name(data_name, batch_size=32, seed=42, layer_indices=(22,), unit_mm=1.0):
    return f"{som_name(layer_indices, unit_mm)}_{data_name}_bs{batch_size}_sd{seed}.pt"


def som_grid_shape(layer_indices, tissue_size_mm=70.0, unit_mm=1.0):
    height = int(round(tissue_size_mm / unit_mm))
    width = int(round(tissue_size_mm / unit_mm))
    return height, width


def make_som_positions(name, layer_indices, tissue_size_mm=70.0, unit_mm=1.0):
    height, width = som_grid_shape(layer_indices, tissue_size_mm=tissue_size_mm, unit_mm=unit_mm)
    xs = (np.arange(width, dtype=np.float32) + 0.5) * unit_mm
    ys = (np.arange(height, dtype=np.float32) + 0.5) * unit_mm
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    coordinates = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    return LayerPositions(
        name=name,
        dims=(1, height, width),
        coordinates=coordinates,
        neighborhood_indices=np.zeros((1, 1), dtype=np.int64),
        neighborhood_width=math.inf,
    )


@torch.no_grad()
def initialize_weights_from_samples(weights, samples, generator=None):
    if samples.shape[0] == 0:
        raise ValueError("Cannot initialize SOM weights from an empty sample set.")

    sample_indices = torch.randint(
        samples.shape[0],
        (weights.shape[0],),
        device=samples.device,
        generator=generator,
    )
    weights.copy_(samples[sample_indices])
    weights.div_(weights.norm(dim=1, keepdim=True).clamp_min(1e-6))


@torch.no_grad()
def som_batch_update(weights, samples, grid_coordinates, lr, sigma, chunk_size=512):
    """Apply one mini-batch SOM update using cosine BMUs and Gaussian grid neighborhoods."""
    if samples.shape[0] == 0:
        return {"quantization_error": float("nan"), "active_units": 0}

    samples = torch.nn.functional.normalize(samples, dim=1)
    normalized_weights = torch.nn.functional.normalize(weights, dim=1)

    numerator = torch.zeros_like(weights)
    denominator = torch.zeros(weights.shape[0], device=weights.device, dtype=weights.dtype)
    bmu_similarities = []
    bmu_index_chunks = []
    sigma2 = max(float(sigma) ** 2, 1e-6)

    for start in range(0, samples.shape[0], chunk_size):
        chunk = samples[start : start + chunk_size]
        similarities, bmu_indices = torch.matmul(chunk, normalized_weights.t()).max(dim=1)
        bmu_similarities.append(similarities)
        bmu_index_chunks.append(bmu_indices)
        bmu_coordinates = grid_coordinates[bmu_indices]
        grid_d2 = (grid_coordinates.unsqueeze(0) - bmu_coordinates.unsqueeze(1)).pow(2).sum(dim=2)
        influence = torch.exp(-grid_d2 / (2.0 * sigma2))
        numerator.add_(torch.matmul(influence.t(), chunk))
        denominator.add_(influence.sum(dim=0))

    active = denominator > 0
    targets = numerator[active] / denominator[active].unsqueeze(1)
    weights[active] += lr * (targets - weights[active])
    weights.div_(weights.norm(dim=1, keepdim=True).clamp_min(1e-6))

    similarities = torch.cat(bmu_similarities)
    bmu_indices = torch.cat(bmu_index_chunks)
    return {
        "quantization_error": (1.0 - similarities).mean().item(),
        "active_units": bmu_indices.unique().numel(),
    }
