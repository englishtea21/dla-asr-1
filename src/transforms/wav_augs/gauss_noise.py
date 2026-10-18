from torch import Tensor, nn, randn_like, rand
import typing
import random
import torchaudio


class AddGaussNoise(nn.Module):
    def __init__(self, prob: float, snr_db_range: Tensor):
        super().__init__()
        self.prob = prob
        self.snr_db_range = snr_db_range

    def forward(self, waveform: Tensor) -> Tensor:
        # waveform: [1, T]
        if random.random() > self.prob:
            return waveform

        noise = randn_like(waveform)
        snr_db = rand(1).to(self.snr_db_range) * (self.snr_db_range[1] - self.snr_db_range[0]) + self.snr_db_range[0]

        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        scale = (signal_power / (noise_power * 10 ** (snr_db.item() / 10))).sqrt()

        return waveform + scale * noise
