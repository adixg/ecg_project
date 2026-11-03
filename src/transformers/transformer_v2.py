"""Convolutional Transformer, compatible with the v2 notebook weights."""

import math
import torch
from torch import nn

from .data import CLASSES, N_LEADS, SIGNAL_LENGTH


class ECGTransformerV2(nn.Module):
    """Convolutional temporal tokens with fixed sinusoidal positions."""

    def __init__(self, lead_mean, lead_std, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        if d_model % n_heads or d_model % 2:
            raise ValueError('d_model must be even and divisible by n_heads.')
        self.register_buffer('lead_mean', torch.as_tensor(lead_mean).reshape(1, 1, N_LEADS))
        self.register_buffer('lead_std', torch.as_tensor(lead_std).reshape(1, 1, N_LEADS))
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=5, padding=2, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.n_tokens = 50
        positions = torch.arange(self.n_tokens, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(self.n_tokens, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequency)
        encoding[:, 1::2] = torch.cos(positions * frequency)
        self.register_buffer('position_encoding', encoding.unsqueeze(0))
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, len(CLASSES))

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (SIGNAL_LENGTH, N_LEADS):
            raise ValueError(f'Expected (batch, {SIGNAL_LENGTH}, {N_LEADS}), got {tuple(x.shape)}')
        x = ((x - self.lead_mean) / self.lead_std).transpose(1, 2)
        tokens = self.stem(x).transpose(1, 2)
        tokens = self.embedding_dropout(self.embedding_norm(tokens) + self.position_encoding)
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        return self.classifier(self.output_norm(tokens).mean(dim=1))
