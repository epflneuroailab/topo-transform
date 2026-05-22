import torch
from torch import nn
import numpy as np


VJEPA_LAYERS = [f'backbone.blocks.{i}' for i in range(24)]
CLIP_LAYERS = [f"vision_model.encoder.layers.{i}" for i in range(24)]
VIDEOMAE_LAYERS = [f"encoder.layer.{i}" for i in range(24)]
UNIFORMER_LAYERS = [
    *[f'model.blocks1.{i}' for i in range(0,  5)],
    *[f'model.blocks2.{i}' for i in range(0,  8)],
    *[f'model.blocks3.{i}' for i in range(0,  20)],
    *[f'model.blocks4.{i}' for i in range(0,  7)],
]


def resolve_sequential_module_from_str(model: nn.Module, model_layer: str) -> nn.Module:
    """
    Recursively resolves the model layer name by drilling into nested nn.Sequential
    """
    # initialize top level as the model
    layer_module = model

    # iterate over parts separated by a period, replacing layer_module with the next
    # sublayer in the chain
    for part in model_layer.split("."):
        layer_module = layer_module._modules.get(part)
        assert (
            layer_module is not None
        ), f"No submodule found for layer {model_layer}, at part {part}."

    return layer_module


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, model, inputs):
        raise NotImplementedError

    def __call__(self, model, inputs):
        model.eval()
        with torch.no_grad():
            features = self.extract_features(model, inputs)
        return features

class LayerFeatureExtractor(FeatureExtractor):
    def __init__(self, layer_names):
        super(LayerFeatureExtractor, self).__init__()
        self.set_layer_names(layer_names)
        self.outputs = {}
        self.hooks = []

    def set_layer_names(self, layer_names):
        self.layer_names = layer_names

    def _get_layer(self, model, layer_name):
        return resolve_sequential_module_from_str(model, layer_name)

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            self.outputs[layer_name] = output
        return hook

    def register_hooks(self, model):
        for layer_name in self.layer_names:
            layer = self._get_layer(model, layer_name)
            hook = layer.register_forward_hook(self._hook_fn(layer_name))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_features(self, model, inputs):
        self.register_hooks(model)
        model(inputs)
        self.remove_hooks()
        features = [self.outputs[name] for name in self.layer_names]
        return features

class VJEPAFeatureExtractor(LayerFeatureExtractor):
    def __init__(self, layer_names=None, layer_indices=None, ret_type='tchw'):
        if layer_names is None:
            if layer_indices is not None:
                layer_names = [VJEPA_LAYERS[i] for i in layer_indices]
            else:
                layer_names = VJEPA_LAYERS

        super().__init__(layer_names)
        self.ret_type = ret_type

    @property
    def layer_dims(self):
        return [(1024, 14, 14) for _ in range(self.num_target_layers)]

    @property
    def num_target_layers(self):
        return len(self.layer_names)

    def extract_features(self, model, inputs):
        if inputs.ndim == 4:
            # img to single-frame video
            inputs = inputs[:,None].repeat(1,2,1,1,1)

        features = super().extract_features(model, inputs)
        features = [self._process_feature(feat) for feat in features]
        return features

    def _process_feature(self, feature):
        # feature: B x THW x C
        if isinstance(feature, tuple):
            feature = feature[0]  # in case of (feature, extra)
        B, THW, C = feature.shape
        feature = feature.reshape(B, -1, 14, 14, C)  # B x T x H x W x C
        feature = feature.permute(0, 1, 4, 2, 3)  # B x T x C x H x W
        if self.ret_type == 'tchw':
            return feature
        elif self.ret_type == 'chw':
            feature = feature.mean(dim=1)  # B x C x H x W


class CLIPFeatureExtractor(FeatureExtractor):
    def __init__(self, layer_indices=None, model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.layer_indices = list(layer_indices or [14, 18, 22])
        self.layer_names = [CLIP_LAYERS[index] for index in self.layer_indices]
        self.model_name = model_name

    @property
    def layer_dims(self):
        return [(1024, 16, 16) for _ in range(self.num_target_layers)]

    @property
    def num_target_layers(self):
        return len(self.layer_indices)

    def set_layer_names(self, layer_names):
        self.layer_names = layer_names
        self.layer_indices = [CLIP_LAYERS.index(name) for name in layer_names]

    def extract_features(self, model, inputs):
        temporal = inputs.ndim == 5
        if temporal:
            bsz, time, channels, height, width = inputs.shape
            inputs = inputs.reshape(bsz * time, channels, height, width)
        else:
            bsz = inputs.shape[0]
            time = 1

        outputs = model(inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        features = [self._process_feature(hidden_states[index + 1], bsz, time) for index in self.layer_indices]
        return features

    def _process_feature(self, feature, bsz, time):
        # Drop CLS token and restore CLIP ViT-L/14's 16 x 16 patch grid.
        feature = feature[:, 1:, :]
        _, tokens, channels = feature.shape
        side = int(tokens ** 0.5)
        assert side * side == tokens, f"Expected square CLIP patch grid, got {tokens} tokens."
        feature = feature.reshape(bsz, time, side, side, channels)
        return feature.permute(0, 1, 4, 2, 3)


class VideoMAEFeatureExtractor(FeatureExtractor):
    def __init__(self, layer_indices=None, model_name="MCG-NJU/videomae-large-finetuned-kinetics"):
        super().__init__()
        self.layer_indices = list(layer_indices or [14, 18, 22])
        self.layer_names = [VIDEOMAE_LAYERS[index] for index in self.layer_indices]
        self.model_name = model_name
        self.hidden_size = 1024
        self.spatial_size = 14
        self.temporal_size = 8

    @property
    def layer_dims(self):
        return [(self.hidden_size, self.spatial_size, self.spatial_size) for _ in range(self.num_target_layers)]

    @property
    def num_target_layers(self):
        return len(self.layer_indices)

    def set_layer_names(self, layer_names):
        self.layer_names = layer_names
        self.layer_indices = [VIDEOMAE_LAYERS.index(name) for name in layer_names]

    def extract_features(self, model, inputs):
        if inputs.ndim != 5:
            inputs = inputs[:, None]
        inputs = self._match_num_frames(inputs, model.config.num_frames)
        outputs = model(inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        self.hidden_size = model.config.hidden_size
        self.spatial_size = model.config.image_size // model.config.patch_size
        self.temporal_size = model.config.num_frames // model.config.tubelet_size
        return [self._process_feature(hidden_states[index + 1]) for index in self.layer_indices]

    @staticmethod
    def _match_num_frames(inputs, num_frames):
        time = inputs.shape[1]
        if time == num_frames:
            return inputs
        indices = torch.linspace(0, time - 1, steps=num_frames, device=inputs.device).round().long()
        return inputs.index_select(1, indices)

    def _process_feature(self, feature):
        bsz, tokens, channels = feature.shape
        expected_tokens = self.temporal_size * self.spatial_size * self.spatial_size
        assert tokens == expected_tokens, f"Expected {expected_tokens} VideoMAE tokens, got {tokens}."
        feature = feature.reshape(bsz, self.temporal_size, self.spatial_size, self.spatial_size, channels)
        return feature.permute(0, 1, 4, 2, 3)


class TDANNFeatureExtractor(LayerFeatureExtractor):
    def __init__(self):
        super().__init__(["model.layer4.1"])

    @property
    def layer_dims(self):
        return [(512, 7, 7)]

    @property
    def num_target_layers(self):
        return 1

    def extract_features(self, model, inputs):
        temporal = inputs.ndim == 5
        if temporal:
            # process frames individually
            b, t, c, h, w = inputs.shape
            inputs = inputs.view(b * t, c, h, w)
            features = super().extract_features(model, inputs)
            features = [feat.view(b, t, *feat.shape[1:]) for feat in features]
        else:
            features = super().extract_features(model, inputs)
        return features
