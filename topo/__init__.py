import os
import numpy as np
import torch
from torch import nn
from contextlib import contextmanager

from config import CACHE_DIR, POSITION_DIR
from .positions import create_position_dicts
from .tissue import _get_tissue_configs_v3, _get_tissue_configs, VJEPA_LAYER_ASSIGNMENTS
from .loss import *
from .smoothing import NeuronSmoothing
from .som import make_som_positions, som_grid_shape, som_name
from spacetorch.models.positions import LayerPositions


def _freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _bufferize_layer_positions(module, layer_positions):
    for index, layer_position in enumerate(layer_positions):
        coords = torch.from_numpy(layer_position.coordinates).float()
        neigh = torch.from_numpy(layer_position.neighborhood_indices).long()
        module.register_buffer(f"layer_{index}_coordinates", coords)
        module.register_buffer(f"layer_{index}_neighborhood_indices", neigh)
        layer_position.coordinates = coords
        layer_position.neighborhood_indices = neigh


def _vjepa_name(
    layer_indices,
    exponentially_interpolate=False,
    constant_rf_overlap=False,
    single_sheet=True,
    large_neighborhood=False,
    inf_neighborhood=True,
    tissue_config="vtc",
    rf_overlap_override=None,
):
    name = "vjepa"
    if list(layer_indices) == list(range(24)):
        name += "_full"
    else:
        name += "".join(f"_{layer_index}" for layer_index in layer_indices)
    if exponentially_interpolate:
        name += "_interp"
    if constant_rf_overlap:
        name += "_constRF"
    if single_sheet:
        name += "_single"
    if large_neighborhood:
        name += "_neighbL"
    if inf_neighborhood:
        name += "_neighbInf"
    if tissue_config != "vtc":
        name += f"_{tissue_config}"
    if rf_overlap_override is not None:
        overlap_str = f"{rf_overlap_override:g}".replace(".", "p")
        name += f"_rf{overlap_str}"
    return name


def _clip_name(layer_indices, single_sheet=True, inf_neighborhood=True, tissue_config="vtc", rf_overlap_override=None):
    name = "clip"
    name += "".join(f"_{layer_index}" for layer_index in layer_indices)
    if single_sheet:
        name += "_single"
    if inf_neighborhood:
        name += "_neighbInf"
    if tissue_config != "vtc":
        name += f"_{tissue_config}"
    if rf_overlap_override is not None:
        overlap_str = f"{rf_overlap_override:g}".replace(".", "p")
        name += f"_rf{overlap_str}"
    return name


def _videomae_name(layer_indices, single_sheet=True, inf_neighborhood=True, tissue_config="vtc", rf_overlap_override=None):
    name = "videomae"
    name += "".join(f"_{layer_index}" for layer_index in layer_indices)
    if single_sheet:
        name += "_single"
    if inf_neighborhood:
        name += "_neighbInf"
    if tissue_config != "vtc":
        name += f"_{tissue_config}"
    if rf_overlap_override is not None:
        overlap_str = f"{rf_overlap_override:g}".replace(".", "p")
        name += f"_rf{overlap_str}"
    return name


def _toponets_name(layer_name="layer4.1", tau=10.0):
    layer_tag = layer_name.replace(".", "_")
    tau_tag = f"{float(tau):g}".replace(".", "p")
    return f"toponets_resnet18_tau{tau_tag}_{layer_tag}_channel_sheet"


def _llcnn_name(layer_name="layer4.1", single_sheet=True, inf_neighborhood=True):
    name = f"llcnn_{layer_name.replace('.', '_')}_channel_sheet"
    if single_sheet:
        name += "_single"
    if inf_neighborhood:
        name += "_neighbInf"
    return name


def _channel_sheet_positions(name, dims, tissue_width_mm=70.0):
    channels, height, width = dims
    assert channels == 1, f"Expected one channel sheet, got dims={dims}."
    unit_mm = tissue_width_mm / max(width - 1, 1)
    coordinates = []
    for y in range(height):
        for x in range(width):
            coordinates.append([x * unit_mm, y * unit_mm])
    coordinates = np.asarray(coordinates, dtype=np.float32)
    indices = np.arange(len(coordinates), dtype=np.int64)
    neighborhood_indices = np.tile(indices[None, :], (len(indices), 1))
    return LayerPositions(
        name=name,
        dims=dims,
        coordinates=coordinates,
        neighborhood_indices=neighborhood_indices,
        neighborhood_width=np.inf,
    )


def _resolve_tissue_configs(
    layer_indices,
    tissue_config="vtc",
    exponentially_interpolate=False,
    constant_rf_overlap=False,
    large_neighborhood=False,
    inf_neighborhood=True,
    rf_overlap_override=None,
):
    if tissue_config == "vtc":
        configs = _get_tissue_configs_v3(
            layer_indices,
            layer_assignments=VJEPA_LAYER_ASSIGNMENTS,
            exponentially_interpolate=exponentially_interpolate,
            constant_rf_overlap=constant_rf_overlap,
            large_neighborhood=large_neighborhood,
            inf_neighborhood=inf_neighborhood,
        )
    elif tissue_config == "small":
        configs = _get_tissue_configs(
            layer_indices,
            layer_assignments=VJEPA_LAYER_ASSIGNMENTS,
            exponentially_interpolate=exponentially_interpolate,
            constant_rf_overlap=constant_rf_overlap,
            large_neighborhood=large_neighborhood,
        )
    else:
        raise ValueError(f"Unknown tissue_config: {tissue_config}")

    layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps = configs
    if rf_overlap_override is not None:
        layer_rf_overlaps = [rf_overlap_override for _ in layer_rf_overlaps]
        print("Layer RF overlaps overridden:", layer_rf_overlaps)
    return layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps


def _load_single_sheet_positions(extractor, file_path):
    assert file_path.exists()
    return [LayerPositions.load(file_path) for _ in extractor.layer_names]


def _load_multi_layer_positions(extractor, layer_config_dir):
    layer_positions = []
    for layer_name in extractor.layer_names:
        file_path = layer_config_dir / f"{layer_name}.npz"
        assert file_path.exists()
        layer_positions.append(LayerPositions.load(file_path))
    return layer_positions


def _layer_config_dict(extractor, values):
    return {name: value for name, value in zip(extractor.layer_names, values)}


class TopoTransformedModel(nn.Module):
    def __init__(self, name, model, extractor, layer_positions, transform, rebuild=False, seed=42):
        super().__init__()

        self.model = _freeze_model(model)

        self.name = name
        self.transform = transform
        self.set_extractor(extractor)
        self.output_layer_indices = range(self.num_layers)
        self.layer_positions = layer_positions

        # smoothing
        self.smoothing = False
        self.fwhm_mm = 2
        self.resolution_mm = 1

        _bufferize_layer_positions(self, self.layer_positions)

    def set_extractor(self, extractor):
        self.extractor = extractor
        self.layer_names = extractor.layer_names
        self.layer_dims = extractor.layer_dims
        self.num_layers = extractor.num_target_layers

    def set_layer_names(self, layer_names):
        self.output_layer_indices = [self.layer_names.index(name) for name in layer_names]
        self.extractor.set_layer_names(layer_names)
        self.set_extractor(self.extractor)

    def smooth(self, layer_features, layer_positions):
        smoothing = NeuronSmoothing(fwhm_mm=self.fwhm_mm, resolution_mm=self.resolution_mm)
        ret_features = []
        ret_positions = []
        for layer_feat, layer_pos in zip(layer_features, layer_positions):
            positions = layer_pos.coordinates
            features_smoothed, positions_smoothed, grid_dims = smoothing(layer_feat, positions)
            layer_pos_smoothed = LayerPositions(
                name=layer_pos.name,
                dims=grid_dims,
                coordinates=positions_smoothed,
                neighborhood_indices=None,
                neighborhood_width=None,
            )
            ret_features.append(features_smoothed)
            ret_positions.append(layer_pos_smoothed)
        return ret_features, ret_positions 

    @contextmanager
    def smoothing_enabled(self, fwhm_mm=None, resolution_mm=None):
        """Context manager to temporarily enable smoothing."""
        original_smoothing = self.smoothing
        old_fwhm = self.fwhm_mm
        old_resolution = self.resolution_mm
        try:
            if resolution_mm == 0.0:
                self.smoothing = False
                self.smoothed_layer_positions = self.layer_positions
                self.fwhm_mm = None
                self.resolution_mm = None
            else:
                self.smoothing = True
                if fwhm_mm is not None:
                    self.fwhm_mm = fwhm_mm
                if resolution_mm is not None:
                    self.resolution_mm = resolution_mm

                self.smoothed_layer_positions = []
                for layer_position in self.layer_positions:
                    position_smoothed, grid_dims = NeuronSmoothing.get_grid_positions(layer_position.coordinates, resolution_mm=resolution_mm)
                    self.smoothed_layer_positions.append(
                        LayerPositions(
                            name=layer_position.name,
                            dims=grid_dims,
                            coordinates=position_smoothed,
                            neighborhood_indices=None,  # to be computed later if needed
                            neighborhood_width=layer_position.neighborhood_width,
                        )
                    )

                    if hasattr(self, "single_sheet") and self.single_sheet:
                        self.smoothed_layer_positions = self.smoothed_layer_positions[:1]

            yield self
        finally:
            self.smoothing = original_smoothing
            self.fwhm_mm = old_fwhm
            self.resolution_mm = old_resolution
            del self.smoothed_layer_positions

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)
        if do_transform:
            layer_features = self.transform(layer_features)
        layer_positions = [self.layer_positions[i] for i in self.output_layer_indices]

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions


class TopoTransformedVJEPA(TopoTransformedModel):
    def __init__(self, layer_indices=[14,18,22], smoothing=False, exponentially_interpolate=False, no_transform=False, 
                constant_rf_overlap=False, rebuild=False, single_sheet=True, large_neighborhood=False, inf_neighborhood=True,
                tissue_config="vtc", rf_overlap_override=None, seed=42, swapopt=False):
        from models import VJEPA, VJEPASwapopt
        from .features import VJEPAFeatureExtractor
        from .layer import TopoTransform

        name = _vjepa_name(
            layer_indices,
            exponentially_interpolate=exponentially_interpolate,
            constant_rf_overlap=constant_rf_overlap,
            single_sheet=single_sheet,
            large_neighborhood=large_neighborhood,
            inf_neighborhood=inf_neighborhood,
            tissue_config=tissue_config,
            rf_overlap_override=rf_overlap_override,
        )

        self.single_sheet = single_sheet
        self.smoothing = smoothing
        self.no_transform = no_transform
        
        self.swapopt = swapopt
        model = VJEPA() if not swapopt else VJEPASwapopt()
        extractor = VJEPAFeatureExtractor(layer_indices=layer_indices)
        transform = TopoTransform(layer_dims=extractor.layer_dims)

        layer_config_dir = (POSITION_DIR / f"{name}_sd{seed}")
        print(layer_config_dir)

        if (not layer_config_dir.exists() or rebuild) and not swapopt:
            print("Generating layer positions...")
            layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps = _resolve_tissue_configs(
                layer_indices,
                tissue_config=tissue_config if single_sheet else "small",
                exponentially_interpolate=exponentially_interpolate,
                constant_rf_overlap=constant_rf_overlap,
                large_neighborhood=large_neighborhood,
                inf_neighborhood=inf_neighborhood,
                rf_overlap_override=rf_overlap_override,
            )

            layer_dims = _layer_config_dict(extractor, extractor.layer_dims)
            layer_tissue_sizes = _layer_config_dict(extractor, layer_tissue_sizes)
            layer_neighborhood_widths = _layer_config_dict(extractor, layer_neighborhood_widths)
            layer_rf_overlaps = _layer_config_dict(extractor, layer_rf_overlaps)

            np.random.seed(seed)
            layer_positions = create_position_dicts(
                extractor.layer_names,
                layer_dims,
                layer_tissue_sizes, 
                layer_neighborhood_widths,
                layer_rf_overlaps,
                save_dir=layer_config_dir,
                single_sheet=single_sheet,
                inf_neighborhood=inf_neighborhood,
            )
        else:
            print(f"Loading layer positions from {layer_config_dir} ...")
            if not single_sheet:
                layer_positions = _load_multi_layer_positions(extractor, layer_config_dir)
            elif swapopt:
                file_path = POSITION_DIR / "swapopt_single_sheet" / f"seed{seed}" / "single_sheet.npz"
                layer_positions = _load_single_sheet_positions(extractor, file_path)
            else:
                file_path = layer_config_dir / "single_sheet.pkl"
                layer_positions = _load_single_sheet_positions(extractor, file_path)

        super().__init__(name, model, extractor, layer_positions, transform, rebuild, seed)

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)

        if do_transform and not self.swapopt and not self.no_transform:
            layer_features = self.transform(layer_features)

        if self.single_sheet:
            # concatenate features along width
            concatenated_features = []
            for feat in layer_features:
                concatenated_features.append(feat)  # list of (B, T, C, H, W)
            if self.swapopt:
                # for swapopt, features are (B, T, C, H, W)
                concatenated_features = torch.cat(concatenated_features, dim=-1)  # (B, T, C, H, W*num_layers)
            else:
                concatenated_features = torch.cat(concatenated_features, dim=2)  # (B, T, C*num_layers, H, W)
            layer_features = [concatenated_features]
            layer_positions = self.layer_positions
        else:
            layer_features = [self.layer_features[i] for i in self.output_layer_indices]

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions
    

class TopoTransformedCLIP(TopoTransformedModel):
    def __init__(
        self,
        layer_indices=[14, 18, 22],
        no_transform=False,
        rebuild=False,
        single_sheet=True,
        inf_neighborhood=True,
        tissue_config="vtc",
        rf_overlap_override=None,
        seed=42,
        clip_model_name="openai/clip-vit-large-patch14",
    ):
        from models import CLIPVision
        from .features import CLIPFeatureExtractor
        from .layer import TopoTransform

        name = _clip_name(
            layer_indices,
            single_sheet=single_sheet,
            inf_neighborhood=inf_neighborhood,
            tissue_config=tissue_config,
            rf_overlap_override=rf_overlap_override,
        )
        self.single_sheet = single_sheet
        self.no_transform = no_transform
        self.clip_model_name = clip_model_name

        model = CLIPVision(model_name=clip_model_name)
        extractor = CLIPFeatureExtractor(layer_indices=layer_indices, model_name=clip_model_name)
        transform = TopoTransform(layer_dims=extractor.layer_dims)

        layer_config_dir = POSITION_DIR / f"{name}_sd{seed}"
        if not layer_config_dir.exists() or rebuild:
            print("Generating CLIP layer positions...")
            layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps = _resolve_tissue_configs(
                layer_indices,
                tissue_config=tissue_config,
                inf_neighborhood=inf_neighborhood,
                rf_overlap_override=rf_overlap_override,
            )
            layer_dims = _layer_config_dict(extractor, extractor.layer_dims)
            layer_tissue_sizes = _layer_config_dict(extractor, layer_tissue_sizes)
            layer_neighborhood_widths = _layer_config_dict(extractor, layer_neighborhood_widths)
            layer_rf_overlaps = _layer_config_dict(extractor, layer_rf_overlaps)

            np.random.seed(seed)
            layer_positions = create_position_dicts(
                extractor.layer_names,
                layer_dims,
                layer_tissue_sizes,
                layer_neighborhood_widths,
                layer_rf_overlaps,
                save_dir=layer_config_dir,
                single_sheet=single_sheet,
                inf_neighborhood=inf_neighborhood,
            )
        else:
            print(f"Loading CLIP layer positions from {layer_config_dir} ...")
            if single_sheet:
                layer_positions = _load_single_sheet_positions(extractor, layer_config_dir / "single_sheet.pkl")
            else:
                layer_positions = _load_multi_layer_positions(extractor, layer_config_dir)

        super().__init__(name, model, extractor, layer_positions, transform, rebuild, seed)

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)

        if do_transform and not self.no_transform:
            layer_features = self.transform(layer_features)

        if self.single_sheet:
            layer_features = [torch.cat(layer_features, dim=2)]
            layer_positions = self.layer_positions
        else:
            layer_features = [layer_features[i] for i in self.output_layer_indices]
            layer_positions = [self.layer_positions[i] for i in self.output_layer_indices]

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions


class TopoTransformedVideoMAE(TopoTransformedModel):
    def __init__(
        self,
        layer_indices=[14, 18, 22],
        no_transform=False,
        rebuild=False,
        single_sheet=True,
        inf_neighborhood=True,
        tissue_config="vtc",
        rf_overlap_override=None,
        seed=42,
        videomae_model_name="MCG-NJU/videomae-large-finetuned-kinetics",
    ):
        from models import VideoMAEVision
        from .features import VideoMAEFeatureExtractor
        from .layer import TopoTransform

        name = _videomae_name(
            layer_indices,
            single_sheet=single_sheet,
            inf_neighborhood=inf_neighborhood,
            tissue_config=tissue_config,
            rf_overlap_override=rf_overlap_override,
        )
        self.single_sheet = single_sheet
        self.no_transform = no_transform
        self.videomae_model_name = videomae_model_name

        model = VideoMAEVision(model_name=videomae_model_name)
        extractor = VideoMAEFeatureExtractor(layer_indices=layer_indices, model_name=videomae_model_name)
        transform = TopoTransform(layer_dims=extractor.layer_dims)

        layer_config_dir = POSITION_DIR / f"{name}_sd{seed}"
        if not layer_config_dir.exists() or rebuild:
            print("Generating VideoMAE layer positions...")
            layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps = _resolve_tissue_configs(
                layer_indices,
                tissue_config=tissue_config,
                inf_neighborhood=inf_neighborhood,
                rf_overlap_override=rf_overlap_override,
            )
            layer_dims = _layer_config_dict(extractor, extractor.layer_dims)
            layer_tissue_sizes = _layer_config_dict(extractor, layer_tissue_sizes)
            layer_neighborhood_widths = _layer_config_dict(extractor, layer_neighborhood_widths)
            layer_rf_overlaps = _layer_config_dict(extractor, layer_rf_overlaps)

            np.random.seed(seed)
            layer_positions = create_position_dicts(
                extractor.layer_names,
                layer_dims,
                layer_tissue_sizes,
                layer_neighborhood_widths,
                layer_rf_overlaps,
                save_dir=layer_config_dir,
                single_sheet=single_sheet,
                inf_neighborhood=inf_neighborhood,
            )
        else:
            print(f"Loading VideoMAE layer positions from {layer_config_dir} ...")
            if single_sheet:
                layer_positions = _load_single_sheet_positions(extractor, layer_config_dir / "single_sheet.pkl")
            else:
                layer_positions = _load_multi_layer_positions(extractor, layer_config_dir)

        super().__init__(name, model, extractor, layer_positions, transform, rebuild, seed)

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)

        if do_transform and not self.no_transform:
            layer_features = self.transform(layer_features)

        if self.single_sheet:
            layer_features = [torch.cat(layer_features, dim=2)]
            layer_positions = self.layer_positions
        else:
            layer_features = [layer_features[i] for i in self.output_layer_indices]
            layer_positions = [self.layer_positions[i] for i in self.output_layer_indices]

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions


class TopoTransformedLLCNN(TopoTransformedModel):
    def __init__(
        self,
        checkpoint_path=None,
        layer_name="layer4.1",
        rebuild=False,
        seed=42,
        inf_neighborhood=True,
    ):
        from models import LLCNNVision
        from .features import LLCNNFeatureExtractor

        self.single_sheet = True
        self.no_transform = True
        self.llcnn_layer_name = layer_name

        name = _llcnn_name(layer_name=layer_name, single_sheet=True, inf_neighborhood=inf_neighborhood)
        model = LLCNNVision(checkpoint_path=checkpoint_path)
        extractor = LLCNNFeatureExtractor(layer_name=layer_name)

        layer_config_dir = POSITION_DIR / f"{name}_sd{seed}"
        if not layer_config_dir.exists() or rebuild:
            layer_dims = _layer_config_dict(extractor, extractor.layer_dims)
            layer_tissue_sizes = _layer_config_dict(extractor, [70.0])
            layer_neighborhood_widths = _layer_config_dict(
                extractor,
                [np.inf if inf_neighborhood else 31.818],
            )
            layer_rf_overlaps = _layer_config_dict(extractor, [1.0])
            np.random.seed(seed)
            layer_positions = create_position_dicts(
                extractor.layer_names,
                layer_dims,
                layer_tissue_sizes,
                layer_neighborhood_widths,
                layer_rf_overlaps,
                save_dir=layer_config_dir,
                single_sheet=True,
                inf_neighborhood=inf_neighborhood,
            )
        else:
            layer_positions = _load_single_sheet_positions(
                extractor,
                layer_config_dir / "single_sheet.pkl",
            )

        super().__init__(name, model, extractor, layer_positions, transform=None, rebuild=rebuild, seed=seed)

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)
        layer_positions = self.layer_positions

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions


class TopoTransformedTopoNets(TopoTransformedModel):
    def __init__(
        self,
        checkpoint_path=None,
        layer_name="layer4.1.conv2",
        tau=10.0,
        seed=42,
        rebuild=False,
    ):
        from models import TopoNetsVision
        from .features import TopoNetsFeatureExtractor

        self.single_sheet = True
        self.no_transform = True
        self.toponets_layer_name = layer_name
        self.toponets_tau = float(tau)

        name = _toponets_name(layer_name=layer_name, tau=tau)
        model = TopoNetsVision(tau=tau, checkpoint_path=checkpoint_path)
        extractor = TopoNetsFeatureExtractor(layer_name=layer_name)
        layer_positions = [
            _channel_sheet_positions(
                extractor.layer_names[0],
                extractor.layer_dims[0],
                tissue_width_mm=70.0,
            )
        ]

        super().__init__(name, model, extractor, layer_positions, transform=None, rebuild=rebuild, seed=seed)

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)
        layer_positions = self.layer_positions

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions


class SOMTopoVJEPA(TopoTransformedModel):
    def __init__(
        self,
        layer_indices=[22],
        tissue_size_mm=70.0,
        unit_mm=1.0,
        seed=42,
        activation_chunk_size=256,
    ):
        from models import VJEPA
        from .features import VJEPAFeatureExtractor

        self.single_sheet = True
        self.tissue_size_mm = tissue_size_mm
        self.unit_mm = unit_mm
        self.activation_chunk_size = activation_chunk_size
        self.grid_shape = som_grid_shape(layer_indices, tissue_size_mm=tissue_size_mm, unit_mm=unit_mm)

        name = som_name(layer_indices, unit_mm=unit_mm)
        model = VJEPA()
        extractor = VJEPAFeatureExtractor(layer_indices=layer_indices)
        layer_positions = [make_som_positions(name, layer_indices, tissue_size_mm=tissue_size_mm, unit_mm=unit_mm)]

        input_dim = extractor.layer_dims[-1][0]
        num_units = self.grid_shape[0] * self.grid_shape[1]
        generator = torch.Generator()
        generator.manual_seed(seed)
        initial_weights = torch.randn(num_units, input_dim, generator=generator)
        initial_weights = torch.nn.functional.normalize(initial_weights, dim=1)

        super().__init__(name, model, extractor, layer_positions, transform=None, rebuild=False, seed=seed)
        self.register_buffer("som_weights", initial_weights)
        self.register_buffer("som_grid_coordinates", self.layer_positions[0].coordinates.clone())

    def extract_som_input_features(self, inputs):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)
        return layer_features[-1].mean(dim=(1, 3, 4))

    @staticmethod
    def flatten_som_input_features(features):
        return features

    def _som_activations(self, features):
        bsz, _ = features.shape
        vectors = self.flatten_som_input_features(features)
        vectors = torch.nn.functional.normalize(vectors, dim=1)
        weights = torch.nn.functional.normalize(self.som_weights, dim=1)

        activation_chunks = []
        for start in range(0, vectors.shape[0], self.activation_chunk_size):
            end = start + self.activation_chunk_size
            chunk = vectors[start:end]
            activation_chunks.append(torch.matmul(chunk, weights.t()))

        activations = torch.cat(activation_chunks, dim=0)
        grid_h, grid_w = self.grid_shape
        return activations.reshape(bsz, 1, 1, grid_h, grid_w)

    def forward(self, inputs, do_transform=True):
        features = self.extract_som_input_features(inputs)
        layer_features = [self._som_activations(features)]
        layer_positions = self.layer_positions

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions


class TopoTransformedTDANN(TopoTransformedModel):
    def __init__(self, seed=0):
        from models import TDANN
        from .features import TDANNFeatureExtractor
        
        
        model = TDANN()
        extractor = TDANNFeatureExtractor()

        name = 'tdann_4.1_single'
        layer_config_dir = (POSITION_DIR / name)
        self.single_sheet = True
        print(layer_config_dir)

        print("Loading layer positions...")
        file_path = layer_config_dir / "single_sheet.npz"
        layer_positions = _load_single_sheet_positions(extractor, file_path)

        # NOTE
        for i in range(len(layer_positions)):
            layer_positions[i].coordinates = layer_positions[i].coordinates * 7  # for some reason, the positions are 7x smaller (cortical size 70mm)

        super().__init__(name, model, extractor, layer_positions, transform=None, rebuild=None, seed=42)

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)

        if do_transform:
            pass

        assert len(layer_features) == 1
        layer_positions = self.layer_positions 
    
        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions
