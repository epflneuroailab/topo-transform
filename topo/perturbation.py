import logging
import operator
from typing import Callable, Tuple

import numpy as np
import scipy.stats
import torch
from torch import nn


def gaussian_spread(mean, cov, locations):
    spread_distribution = scipy.stats.multivariate_normal(mean=mean, cov=cov)
    scale = spread_distribution.pdf(mean)
    pdf = spread_distribution.pdf(locations)
    return pdf / scale


class LayerPerturbation:
    layer_transforms = {}
    def __init__(self, changes, change_function=operator.mul):
        self._changes = changes
        self._change_function = change_function

    def __call__(self, layer_module, input, output):
        output_multiple = False
        if isinstance(output, tuple):
            extra = output[1:]
            output = output[0]
            output_multiple = True

        changes = self._changes
        transform = getattr(layer_module, '_transform', None)
        if transform is None:
            raise AttributeError("Layer module missing _transform; did you call bind_transforms()?")

        # assume VJEPA
        B, L, C = output.shape
        H = getattr(transform, "H", None)
        W = getattr(transform, "W", None)
        if H is None or W is None:
            raise AttributeError("Transform missing H/W; cannot infer spatial dims.")
        if L % (H * W) != 0:
            raise ValueError(f"Output length {L} not divisible by H*W ({H}*{W}).")
        T = L // (H * W)
        output = output.reshape(B, T, H, W, C)  # (B, T, H, W, C)
        output = output.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

        output = output.reshape(-1, C, H, W)  # (B*T, C, H, W)
        result = transform(output)

        if hasattr(result, 'device'):
            changes = torch.Tensor(changes).to(result.device)
        
        result = self._change_function(result, changes)
        stimulated_output = transform.inverse(result)
        stimulated_output = stimulated_output.reshape(B, -1, C, H, W)  # (B, T, C, H, W)

        stimulated_output = stimulated_output.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        stimulated_output = stimulated_output.reshape(B, L, C)  # (B, L, C)
        
        if output_multiple:
            stimulated_output = (stimulated_output, *extra)
            return stimulated_output
        else:
            return stimulated_output


class TopoModelPerturbation:
    def __init__(self, topo_model, identifier_suffix_max_len=3):
        assert hasattr(topo_model.model, 'register_forward_hook'), 'PyTorch model required'
        self.topo_model = topo_model
        self.bind_transforms(topo_model)
        self._base_identifier = topo_model.name
        self._hooks = []
        self._max_len = identifier_suffix_max_len
        self._logger = logging.getLogger(self.__class__.__name__)

    def bind_transforms(self, topo_model):
        transforms = topo_model.transform.transforms
        layers = [self._get_layer_module(name) for name in topo_model.layer_names]
        for layer, transform in zip(layers, transforms):
            layer._transform = transform
        return topo_model

    def perturb(self, location, perturbation_params, layer_idx=None):
        """Apply perturbation to specified layer at given location.
        
        Args:
            location: (x, y) coordinates on tissue map
            perturbation_params: dict with perturbation-specific parameters
            layer_idx: Index of layer to perturb (only for non-single_sheet mode)
        """
        if self.topo_model.single_sheet:
            # All layers share same position map, concatenated along width
            # Changes are distributed across all layers based on spatial position
            layer_pos = self.topo_model.layer_positions[0]
            all_coordinates = layer_pos.coordinates.cpu().numpy()
            dims = layer_pos.dims  # (C_total, H, W)
            C_total, H, W = dims
            num_layers = len(self.topo_model.layer_names)
            C_per_layer = C_total // num_layers
            
            # Compute changes for all concatenated positions
            changes_flat, change_fn = self.compute_changes(
                all_coordinates, 
                location, 
                **perturbation_params
            )

            from matplotlib import pyplot as plt
            plt.scatter(all_coordinates[:,0], all_coordinates[:,1], c=changes_flat, cmap='viridis', s=1)
            plt.colorbar()
            # equally scale x and y axes
            plt.axis('equal')
            plt.savefig('debug_changes_flat.png')
            plt.close()
            
            # Reshape changes from flat to (C_total, H, W)
            changes_concat = changes_flat.reshape(C_total, H, W)
            
            # Distribute changes across all layers
            for idx, layer_name in enumerate(self.topo_model.layer_names):
                # Extract this layer's slice from concatenated width
                c_start = idx * C_per_layer
                c_end = (idx + 1) * C_per_layer
                layer_changes = changes_concat[c_start:c_end, :, :]  # (C, H, W)
                
                perturbation = LayerPerturbation(layer_changes, change_fn)
                layer_module = self._get_layer_module(layer_name)
                hook = layer_module.register_forward_hook(perturbation)
                self._hooks.append(hook)
            
            identifier_layer = "all"
        else:
            # Non-single_sheet: perturb specific layer
            if layer_idx is None:
                raise ValueError("layer_idx must be specified for non-single_sheet mode")
            
            layer_pos = self.topo_model.layer_positions[layer_idx]
            changes, change_fn = self.compute_changes(
                layer_pos.coordinates.cpu().numpy(), 
                location, 
                **perturbation_params
            )
            if isinstance(changes, np.ndarray) and changes.ndim == 1:
                C, H, W = layer_pos.dims
                changes = changes.reshape(C, H, W)
            layer_name = self.topo_model.layer_names[layer_idx]
            layer_module = self._get_layer_module(layer_name)
            identifier_layer = layer_name
            
            perturbation = LayerPerturbation(changes, change_fn)
            hook = layer_module.register_forward_hook(perturbation)
            self._hooks.append(hook)
        
    def _get_layer_module(self, layer_name):
        """Get the actual PyTorch module for a layer name."""
        parts = layer_name.split('.')
        module = self.topo_model.model
        for part in parts:
            module = getattr(module, part)
        return module

    def _abbrev(self, key: str) -> str:
        return '_'.join([p[:self._max_len] for p in key.split('_')])

    def compute_changes(self, neuron_locs, location, **kwargs) -> Tuple[np.array, Callable]:
        raise NotImplementedError()

    def clear(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class MuscimolInjection(TopoModelPerturbation):
    def __init__(self, topo_model, cov=1.17):
        super().__init__(topo_model)
        self._cov = cov

    def compute_changes(self, neuron_locs, location, amount_microliter, **kwargs):
        assert amount_microliter == 1, "Effects based on 1μL (Arikan et al. 2002)"
        magnitude = amount_microliter
        spread = gaussian_spread(location, self._cov, neuron_locs)
        changes = np.maximum(1 - (magnitude * spread), 0)
        return changes, operator.mul


class OptogeneticSuppression(TopoModelPerturbation):
    def __init__(self, topo_model, cov=0.234):
        super().__init__(topo_model)
        self._cov = cov

    def compute_changes(self, neuron_locs, location, fiber_output_power_mW, **kwargs):
        magnitude = fiber_output_power_mW  # ≥1 mW → 100% suppression (Chow et al. 2010)
        spread = gaussian_spread(location, self._cov, neuron_locs)
        changes = np.maximum(1 - (magnitude * spread), 0)
        return changes, operator.mul


class UnitAblation(TopoModelPerturbation):
    def compute_changes(self, neuron_locs, location, ablation_radius_mm, **kwargs):
        distances = np.linalg.norm(neuron_locs - location, axis=1)
        changes = np.ones_like(distances, dtype=float)
        changes[distances <= ablation_radius_mm] = 0.0
        return changes, operator.mul


class MicroStimulation(TopoModelPerturbation):
    BASELINE_HZ = 30
    MAXIMUM_HZ = 60

    def compute_changes(self, neuron_locs, location, current_pulse_mA, pulse_rate_Hz, **kwargs):
        distances = np.linalg.norm(neuron_locs - location, axis=1)
        # Exponential fit from Kumaravelu et al. 2021
        scale = np.exp(-distances / (0.00505047368 * (current_pulse_mA + -5.26062113)))
        scale *= pulse_rate_Hz / self.BASELINE_HZ
        scale += 1
        # scale = np.minimum(scale, self.MAXIMUM_HZ / self.BASELINE_HZ)
        return scale, operator.mul
