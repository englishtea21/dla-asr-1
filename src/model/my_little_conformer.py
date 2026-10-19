"""
Тини литл конформер. Увы - не стриминговый.
В основу брал https://www.arxiv.org/pdf/2005.08100
"""

from typing import Any

from torch import nn
import torch

import torch.nn.functional as F


class ConvSubsampler(nn.Module):
    """
    Causal temporal convolution with stride=2.

    Input:  [B, mel_banks, T]
    Output: [B, mel_banks, ceil(T / 2)]
    """

    def __init__(
        self,
        mel_banks: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")

        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=mel_banks,
            out_channels=mel_banks,
            kernel_size=kernel_size,
            stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, mel_banks, T]
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x)

    def transform_input_lengths(
        self,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return (lengths + 1) // 2


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module.

    Input:  [B, T, dim]
    Output: [B, T, dim]
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")

        self.norm = nn.LayerNorm(dim)

        self.pointwise_in = nn.Conv1d(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=1,
        )

        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=0,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()

        self.pointwise_out = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout_p)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, dim]
        residual = x

        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, dim, T]

        x = self.pointwise_in(x)  # [B, 2 * dim, T]
        x = F.glu(x, dim=1)  # [B, dim, T]

        # для сохранения временной размерности
        left_pad = (self.kernel_size - 1) // 2
        right_pad = self.kernel_size - 1 - left_pad
        x = F.pad(x, (left_pad, right_pad))

        x = self.depthwise(x)  # [B, dim, T]
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_out(x)  # [B, dim, T]

        x = x.transpose(1, 2)  # [B, T, dim]
        return residual + self.dropout(x)


class FNNBlock(nn.Module):
    def __init__(self, dim: int, dropout_p: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.exp_linear = nn.Linear(dim, dim * 4)
        self.swish_activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_p)
        self.proj_linear = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, dim, T]
        y = self.norm(x)
        y = self.exp_linear(y)
        y = self.swish_activation(y)
        y = self.dropout(y)
        y = self.proj_linear(y)
        y = self.dropout(y)
        return x + y


# взял с https://machinelearningmastery.com/positional-encodings-in-transformer-models/
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        N = 10000
        inv_freq = 1.0 / (N ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    @staticmethod
    def _apply_rotary_pos_emb(x, cos, sin):
        def _rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        return (x * cos) + (_rotate_half(x) * sin)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return self._apply_rotary_pos_emb(x, cos, sin)


# с https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
class RotaryMHSA(nn.Module):
    def __init__(self, dim, n_heads, dropout_p=0.1, max_seq_len=10000):
        super().__init__()

        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout_p = dropout_p

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

        self.rope = RotaryPositionalEncoding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.rope(q, seq_len=T)
        k = self.rope(k, seq_len=T)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class MyConformerBlock(nn.Module):
    def __init__(
        self, n_heads: int, dim: int, dropout_p: float, kernel_size: int
    ) -> None:
        super().__init__()
        self.fnn1 = FNNBlock(dim=dim, dropout_p=dropout_p)
        self.fnn2 = FNNBlock(dim=dim, dropout_p=dropout_p)
        self.rot_mhsa = RotaryMHSA(
            dim=dim, n_heads=n_heads, dropout_p=dropout_p, max_seq_len=100000
        )
        self.conv_block = ConformerConvModule(dim, kernel_size, dropout_p=dropout_p)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, dim, T]
        x = x + 0.5 * self.fnn1(x)
        attn_out = self.rot_mhsa(x)
        x = x + attn_out
        x = x + self.conv_block(x)
        return self.norm(x + 0.5 * self.fnn2(x))


class MyConformer(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        n_blocks: int,
        mel_banks: int,
        dropout_p: float,
        n_heads: int,
        kernel_size_subsample: int,
        kernel_size_conf: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.blocks_count = n_blocks

        self.conv_subsampler = ConvSubsampler(
            mel_banks=mel_banks, kernel_size=kernel_size_subsample
        )
        self.conformer_blocks = nn.Sequential(
            *[
                MyConformerBlock(
                    n_heads=n_heads,
                    dim=mel_banks,
                    dropout_p=dropout_p,
                    kernel_size=kernel_size_conf,
                )
                for _ in range(n_blocks)
            ]
        )
        self.features_to_logits = nn.Linear(
            in_features=mel_banks, out_features=n_tokens
        )

    def forward(
        self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch
    ):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram. [B, n_mels, T]
            spectrogram_length (Tensor): spectrogram original lengths. [B]
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram_subsampled = self.conv_subsampler(spectrogram)  # [B, n_mels, T']
        spectrogram_subsampled_transposed = spectrogram_subsampled.transpose(
            1, 2
        )  # [B, T', dim]
        conf_features = self.conformer_blocks(
            spectrogram_subsampled_transposed
        )  # [B, T', dim]
        output = self.features_to_logits(conf_features)  # [B, T', n_tokens]

        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return self.conv_subsampler.transform_input_lengths(input_lengths)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
