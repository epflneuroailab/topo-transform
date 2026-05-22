import torch
from torch import nn


class CLIPVision(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import CLIPVisionModel

        self.model_name = model_name
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)

    def forward(self, x, output_hidden_states=True):
        return self.vision_model(pixel_values=x, output_hidden_states=output_hidden_states)
