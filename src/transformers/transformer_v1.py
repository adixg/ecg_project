"""Original linear-patch Transformer, compatible with the v1 notebook weights."""

from torch import nn
import torch


class ECGTransformerV1(nn.Module):
    """Linear temporal patches with learned positions; input is (batch, 1000, 12)."""

    def __init__(self, lead_mean, lead_std, signal_length=1000, n_leads=12,
                 n_classes=5, patch_size=20, d_model=128, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        if signal_length % patch_size != 0:
            raise ValueError("Signal length must be divisible by patch size.")
        if d_model % n_heads != 0:
            raise ValueError("Embedding dimension must be divisible by attention heads.")
        self.signal_length = signal_length
        self.n_leads = n_leads
        self.n_patches = signal_length // patch_size
        self.patch_size = patch_size
        # Buffers move with the model and are included in its saved state.
        self.register_buffer("lead_mean", torch.as_tensor(lead_mean).reshape(1, 1, n_leads))
        self.register_buffer("lead_std", torch.as_tensor(lead_std).reshape(1, 1, n_leads))
        self.patch_embedding = nn.Linear(patch_size * n_leads, d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, self.n_patches, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        # Construct each layer independently so initial weights differ by layer.
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, signals):
        if signals.ndim != 3 or signals.shape[1:] != (self.signal_length, self.n_leads):
            raise ValueError(f"Expected (batch, {self.signal_length}, {self.n_leads}).")
        signals = (signals - self.lead_mean) / self.lead_std
        patches = signals.reshape(signals.shape[0], self.n_patches,
                                  self.patch_size * self.n_leads)
        tokens = self.embedding_dropout(self.patch_embedding(patches) + self.position_embedding)
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        pooled = self.output_norm(tokens).mean(dim=1)
        return self.classifier(pooled)
