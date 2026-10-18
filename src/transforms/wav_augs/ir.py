from torch import Tensor, nn, arange
import random
import pathlib
import torchaudio.functional as F


class ImpulseResponseAugmentation(nn.Module):
    """
    Random RIR augmentation using the FLAIR dataset:
    https://zenodo.org/records/17037517

    Args:
        sample_rate: Target sample rate.
        prob: Probability of applying augmentation.
        normalize_rir: Whether to L2-normalize each RIR.
        decay_range: Range of exponential decay coefficients (1/s).
            0.0 means no additional decay.
        mat_rirs_path: Path to the .mat file containing RIRs.
    """

    def __init__(
        self,
        sample_rate: int,
        prob: float = 0.5,
        normalize_rir: bool = True,
        decay_range: tuple[float, float] = (0.0, 3.0),
        mat_rirs_path: pathlib.Path = pathlib.Path(
            "./_data_for_augmentations/data_FLAIR.mat"
        ),
    ):
        super().__init__()

        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")

        min_decay, max_decay = decay_range
        if not 0.0 <= min_decay <= max_decay:
            raise ValueError("decay_range must satisfy 0 <= min <= max")

        self.mat_rirs_path = str(mat_rirs_path)
        self.sample_rate = sample_rate
        self.prob = prob
        self.normalize_rir = normalize_rir
        self.decay_range = decay_range

        self._rirs = Tensor()  # [N, T]
        self._rir_sr = 48000

    def _load_and_prepare_rirs(self, device, dtype):
        """Load, resample and optionally normalize all RIRs once."""
        if self._rirs.numel() > 0:
            return

        from pymatreader import read_mat

        raw_rirs = read_mat(self.mat_rirs_path)["rirs"]
        rirs = Tensor(raw_rirs)[:, :, 0].squeeze(-1).permute(1, 0)

        if self._rir_sr != self.sample_rate:
            rirs = F.resample(
                rirs,
                orig_freq=self._rir_sr,
                new_freq=self.sample_rate,
            )

        if self.normalize_rir:
            rirs = rirs / (rirs.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        self._rirs = rirs.to(device=device, dtype=dtype)

    def _apply_decay(self, rir: Tensor) -> Tensor:
        """Apply random exponential decay to the RIR."""
        decay = random.uniform(*self.decay_range)

        if decay == 0.0:
            return rir

        t = (
            arange(
                rir.size(-1),
                device=rir.device,
                dtype=rir.dtype,
            )
            / self.sample_rate
        )

        envelope = (-decay * t).exp()
        return rir * envelope

    def forward(self, waveform: Tensor) -> Tensor:
        # waveform: [1, T]
        if random.random() >= self.prob:
            return waveform

        self._load_and_prepare_rirs(
            device=waveform.device,
            dtype=waveform.dtype,
        )

        idx = random.randrange(self._rirs.size(0))
        rir = self._rirs[idx].unsqueeze(0)  # [1, T_rir]

        # Modify the impulse response before convolution.
        rir = self._apply_decay(rir)

        output = F.fftconvolve(waveform, rir)[..., : waveform.size(-1)]

        return (output / (output.max().abs() + 1e-8)).clamp(min=-1.0, max=1.0)
