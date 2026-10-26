"""A compact CNN-BiLSTM baseline for 12-lead ECG classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Convolution, normalization, activation, and temporal downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )


class CNNBiLSTM(nn.Module):
    """Three CNN blocks followed by one bidirectional LSTM layer."""

    def __init__(
        self,
        *,
        input_channels: int = 12,
        conv_channels: tuple[int, int, int] = (32, 64, 96),
        lstm_hidden_size: int = 64,
        num_classes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c1, c2, c3 = conv_channels
        self.cnn = nn.Sequential(
            ConvBlock(input_channels, c1, kernel_size=7),
            ConvBlock(c1, c2, kernel_size=5),
            ConvBlock(c2, c3, kernel_size=3),
        )
        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * lstm_hidden_size, num_classes)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Returns one raw logit per diagnostic superclass."""

        features = self.cnn(signal)
        sequence = features.transpose(1, 2)
        sequence, _ = self.lstm(sequence)
        pooled = sequence.mean(dim=1)
        return self.classifier(self.dropout(pooled))
