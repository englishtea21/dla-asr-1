import random

import torch
from torch import Tensor, nn


class AddGaussNoise(nn.Module):
    def __init__(self, prob: float, snr_db_range):
        super().__init__()
        self.prob = prob
        self.snr_db_range = torch.as_tensor(snr_db_range, dtype=torch.float32)
        if self.snr_db_range.numel() != 2:
            raise ValueError("snr_db_range must contain [min, max]")

    def forward(self, waveform: Tensor) -> Tensor:
        if random.random() > self.prob:
            return waveform

        noise = torch.randn_like(waveform)
        snr_db = torch.empty(1, device=waveform.device).uniform_(
            self.snr_db_range[0].item(),
            self.snr_db_range[1].item(),
        )

        signal_power = waveform.pow(2).mean().clamp_min(1e-8)
        noise_power = noise.pow(2).mean().clamp_min(1e-8)
        scale = (signal_power / (noise_power * 10 ** (snr_db / 10))).sqrt()

        return waveform + scale * noise
