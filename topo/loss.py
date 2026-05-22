from torch import nn

from spacetorch.losses.losses_torch import SpatialCorrelationLossModule


class SpatialCorrelationLoss(nn.Module):
    def __init__(self, num_layers: int, neighborhoods_per_batch: int = 16, single_sheet: bool = False):
        super().__init__()
        self.layer_losses = nn.ModuleList([
            SpatialCorrelationLossModule(neighborhoods_per_batch=neighborhoods_per_batch)
            for _ in range(num_layers)
        ])
        self.num_layers = 1 if single_sheet else num_layers
        self.single_sheet = single_sheet

    def forward(self, layer_features, layer_positions):
        loss = 0
        assert len(layer_features) == len(layer_positions) == self.num_layers
            
        for layer_feature, layer_position, layer_loss in zip(layer_features, layer_positions, self.layer_losses):
            has_time = layer_feature.ndim == 5  # BTCHW

            if has_time:
                B, T, C, H, W = layer_feature.shape
                feature = layer_feature.reshape(B*T, C*H*W)
            else:
                B, C, H, W = layer_feature.shape
                feature = layer_feature.reshape(B, C*H*W)

            loss += layer_loss(
                feature,
                layer_position.coordinates,
                layer_position.neighborhood_indices,
            )

        return loss / self.num_layers


import torch
import torch.nn as nn

from spacetorch.utils.torch_utils import corrcoef, pdist, pearsonr, lower_tri

class GlobalSpatialCorrelationLoss(nn.Module):
    def __init__(self, samples_per_batch: int = 8192):
        super().__init__()
        self.samples_per_batch = samples_per_batch

    def forward(self, layer_features, layer_positions):
        layer_feature = layer_features[0]
        layer_position = layer_positions[0]
        coordinates = layer_position.coordinates.to(layer_feature.device)

        has_time = layer_feature.ndim == 5  # BTCHW
        if has_time:
            B, T, C, H, W = layer_feature.shape
            feature = layer_feature.reshape(B*T, C*H*W)
        else:
            B, C, H, W = layer_feature.shape
            feature = layer_feature.reshape(B, C*H*W)

        sampled_indices = torch.randperm(feature.shape[1])[:self.samples_per_batch]
        feature = feature[:, sampled_indices]
        coordinates = coordinates[sampled_indices]

        # compute spatial and repsonse similarity
        distance_similarity = 1 / (pdist(coordinates) + 1)
        response_similarity = corrcoef(feature.t())

        # we want to maximize the alignment between spatial and response similarity
        similarity_alignment = pearsonr(
            lower_tri(response_similarity), lower_tri(distance_similarity)
        )

        # similarity alignment is in [-1, 1], so we convert to a distance by subtracting
        # from 1.0 The value will be in [0, 2], so we divide by 2.0 to guarantee values in
        # [0, 1]
        loss = (1 - similarity_alignment) / 2.0
        return loss
