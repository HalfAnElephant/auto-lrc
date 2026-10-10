"""Conditional TCN boundary heads, separate from the legacy acoustic models."""
import torch
from torch import nn

from .neural_features import BoundaryConfig, PHONE_VOCABULARY, _config


class _ResidualBlock(nn.Module):
    def __init__(self, config, dilation):
        super().__init__()
        channels = config.channels
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, config.kernel_size, padding=dilation, dilation=dilation),
            nn.GroupNorm(config.norm_groups, channels), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Conv1d(channels, channels, config.kernel_size, padding=dilation, dilation=dilation),
            nn.GroupNorm(config.norm_groups, channels), nn.SiLU(), nn.Dropout(config.dropout))

    def forward(self, value, mask):
        return (value + self.layers(value)) * mask


class BoundaryTCN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = _config(config)
        config = self.config
        self.phone_embedding = nn.Embedding(len(PHONE_VOCABULARY), config.embedding_dim)
        self.input_projection = nn.Conv1d(58 + 3 * config.embedding_dim, config.channels, 1)
        self.blocks = nn.ModuleList([_ResidualBlock(config, dilation) for dilation in config.dilations])
        self.heatmap_head = nn.Conv1d(config.channels, 1, 1)
        self.confidence_head = nn.Linear(config.channels, 1)

    def forward(self, features, phones, mask):
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[1:] != (58, self.config.window_frames) or not features.is_floating_point()
                or not bool(torch.isfinite(features).all())):
            raise ValueError("features must be finite floating [B,58,281]")
        batch, _, frames = features.shape
        if (not isinstance(phones, torch.Tensor) or phones.shape != (batch, 3)
                or phones.dtype != torch.long or not isinstance(mask, torch.Tensor)
                or mask.shape != (batch, frames) or mask.dtype != torch.bool
                or phones.device != features.device or mask.device != features.device):
            raise ValueError("phones and mask must be same-device long [B,3] and bool [B,281]")
        if batch == 0 or not bool(mask.any(dim=1).all()) or bool((phones < 0).any()) or bool((phones >= len(PHONE_VOCABULARY)).any()):
            raise ValueError("Every example needs valid positions and in-vocabulary phone IDs")
        if bool((features[:, 57].abs() > 1 + 1e-6).any()):
            raise ValueError("relative_position must be in [-1,1]")
        expanded_mask = mask[:, None].to(features.dtype)
        context = self.phone_embedding(phones).flatten(1)[:, :, None].expand(-1, -1, frames)
        value = self.input_projection(torch.cat([features * expanded_mask, context], dim=1)) * expanded_mask
        for block in self.blocks:
            value = block(value, expanded_mask)
        # Keep heads and softmax in float32 under mixed precision.
        logits = self.heatmap_head(value).squeeze(1).float()
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        mass = torch.softmax(logits, dim=1)
        delta = (mass * features[:, 57].float()).sum(dim=1) * self.config.radius_seconds
        pooled = (value.float() * mass[:, None]).sum(dim=2)
        return {"heatmap_logits": logits, "delta_seconds": delta,
                "confidence_logit": self.confidence_head(pooled).squeeze(1).float()}
