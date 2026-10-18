import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Gain(*args, output_type="tensor", **kwargs)

    def forward(self, waveform: Tensor) -> Tensor:
        # waveform: [1, T]
        waveform = waveform.unsqueeze(0)
        result = self._aug(waveform)
        return result.squeeze(0)
