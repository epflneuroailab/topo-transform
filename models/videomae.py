from torch import nn


class VideoMAEVision(nn.Module):
    def __init__(self, model_name="MCG-NJU/videomae-large-finetuned-kinetics"):
        super().__init__()
        from transformers import VideoMAEModel

        self.model_name = model_name
        self.video_model = VideoMAEModel.from_pretrained(model_name)
        self.config = self.video_model.config

    def forward(self, x, output_hidden_states=True):
        return self.video_model(pixel_values=x, output_hidden_states=output_hidden_states)
