import torch
from torch import nn
import numpy as np


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
        self.layer_names = layer_names
        self.outputs = {}
        self.hooks = []

    def _get_layer(self, model, layer_name):
        modules = layer_name.split('.')
        layer = model
        for mod in modules:
            if mod.isdigit():
                layer = layer[int(mod)]
            else:
                layer = getattr(layer, mod)
        return layer

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
        return features if len(features) > 1 else features[0]

class EvenlySampledLayerExtractor(LayerFeatureExtractor):
    def __init__(self, num_layers):
        # initialize until the model is known
        super(EvenlySampledLayerExtractor, self).__init__(layer_names=[])
        self.num_layers = num_layers
        self.initialized = False

    def extract_features(self, model, inputs):
        if not self.initialized:
            self.layer_names = self._get_evenly_sampled_layers(model, inputs, self.num_layers)
            self.initialized = True
        return super(EvenlySampledLayerExtractor, self).extract_features(model, inputs)

    def _get_evenly_sampled_layers(self, model, inputs, num_layers):
        layers = self.get_execution_order(model, inputs)
        total_layers = len(layers)
        if num_layers > total_layers:
            print(f"Requested {num_layers} layers, but model has only {total_layers} layers. Using all layers.")
            num_layers = total_layers
        step = total_layers / num_layers
        sampled_layers = [layers[int(i * step)][0] for i in range(num_layers)]
        return sampled_layers

    @staticmethod
    def get_execution_order(model, input_tensor):
        """
        Returns a list of (module_name, module) in the order they are executed during a forward pass.
        
        Args:
            model: A torch.nn.Module
            input_tensor: Input tensor(s) for the model
            
        Returns:
            List of (module_name, module) tuples in execution order.
        """
        execution_order = []
        seen = set()
        module_to_name = {module: name for name, module in model.named_modules()}
        
        def hook_fn(module, input, output):
            if module not in seen:
                name = module_to_name.get(module, "")

                # monkey patch
                if ("R3D_ST" in str(model.__class__) or (hasattr(model, "module") and "R3D_ST" in str(model.module.__class__))):
                    if "mixing" in name:
                        return

                execution_order.append((name, module))
                seen.add(module)
        
        hooks = []
        for m in model.modules():
            if m != model:  # skip top-level model itself (optional)
                hooks.append(m.register_forward_hook(hook_fn))
        
        model.eval()
        with torch.no_grad():
            model(input_tensor)
        
        for h in hooks:
            h.remove()
        
        return execution_order