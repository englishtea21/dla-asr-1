import random
import torch
import torchaudio
from torch import Tensor, nn


class RandomPitchShift(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        min_steps: float = -2.0,
        max_steps: float = 2.0,
        prob: float = 0.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.prob = prob

    def forward(self, waveform: Tensor) -> Tensor:
        # waveform: [1, T]
        if random.random() >= self.prob:
            return waveform

        n_steps = int(
            random.uniform(
                self.min_steps,
                self.max_steps,
            )
        )

        waveform = waveform.unsqueeze(0)

        transform = torchaudio.transforms.PitchShift(
            sample_rate=self.sample_rate, n_steps=n_steps,
        ).to(device=waveform.device)

        result = transform(waveform)
        return result.squeeze(0)
