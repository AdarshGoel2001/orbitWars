"""Small edge-token transformer for Orbit Wars.

The harness owns game mechanics and action legality. This module only maps
edge features to a masked policy over edges and a scalar value estimate.
"""

from __future__ import annotations

import torch
from torch import nn

from harness import FEATURE_DIM, FEATURE_SCALES, N_MAX_DEFAULT


class OrbitWarsTransformer(nn.Module):
    """Transformer over `(src, tgt)` edge tokens.

    The policy chooses one `(src, tgt)` edge action or a stop action. Ship
    sizing is deterministic in `agents.py` so the model focuses on target
    selection.
    """

    def __init__(
        self,
        n_max: int = N_MAX_DEFAULT,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        value_hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_max = n_max
        self.feature_dim = feature_dim
        self.d_model = d_model

        self.register_buffer(
            "feature_scales",
            torch.as_tensor(FEATURE_SCALES, dtype=torch.float32).view(1, 1, 1, -1),
        )
        self.edge_embed = nn.Linear(feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, 1)
        self.stop_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, edge_features, legal_mask, action_mask=None):
        """Return masked policy outputs and value.

        Args:
            edge_features: Float tensor shaped `(B, N, N, F)` or `(N, N, F)`.
            legal_mask: Bool tensor shaped `(B, N, N)` or `(N, N)`.
                Cheap candidate mask from GameView feature construction.
            action_mask: Optional bool tensor shaped `(B, N, N)` or `(N, N)`.
                Authoritative radar-validated mask for sampling/training
                policy actions. If omitted, `legal_mask` is used, but normal
                agent/training code should pass GameView.action_mask(...).

        Returns:
            dict with move logits/probabilities, stop probability, flat action
            logits/probabilities, and value.
        """
        if edge_features.dim() == 3:
            edge_features = edge_features.unsqueeze(0)
        if legal_mask.dim() == 2:
            legal_mask = legal_mask.unsqueeze(0)
        if action_mask is not None and action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(0)

        device = self.feature_scales.device
        edge_features = edge_features.to(device=device, dtype=self.feature_scales.dtype)
        legal_mask = legal_mask.to(dtype=torch.bool, device=device)
        if action_mask is not None:
            action_mask = action_mask.to(dtype=torch.bool, device=device)
        scales = self.feature_scales

        batch, n_src, n_tgt, feature_dim = edge_features.shape
        if n_src != self.n_max or n_tgt != self.n_max:
            raise ValueError(f"expected N={self.n_max}, got {(n_src, n_tgt)}")
        if feature_dim != self.feature_dim:
            raise ValueError(f"expected F={self.feature_dim}, got {feature_dim}")
        if legal_mask.shape != (batch, n_src, n_tgt):
            raise ValueError(
                f"legal_mask shape {tuple(legal_mask.shape)} does not match "
                f"{(batch, n_src, n_tgt)}"
            )
        if action_mask is None:
            action_mask = legal_mask
        elif action_mask.shape != (batch, n_src, n_tgt):
            raise ValueError(
                f"action_mask shape {tuple(action_mask.shape)} does not match "
                f"{(batch, n_src, n_tgt)}"
            )

        tokens = (edge_features / scales.clamp_min(1e-6)).reshape(
            batch, n_src * n_tgt, feature_dim
        )
        encoded = self.encoder(self.edge_embed(tokens))

        move_logits = self.policy_head(encoded).squeeze(-1).reshape(batch, n_src, n_tgt)
        pooled = encoded.mean(dim=1)
        stop_logits = self.stop_head(pooled).squeeze(-1)

        flat_move_logits = move_logits.reshape(batch, n_src * n_tgt)
        flat_mask = action_mask.reshape(batch, n_src * n_tgt)
        masked_flat_logits = flat_move_logits.masked_fill(~flat_mask, -1.0e9)
        action_logits = torch.cat(
            [masked_flat_logits, stop_logits.unsqueeze(-1)], dim=-1
        )
        action_policy = torch.softmax(action_logits, dim=-1)
        flat_move_policy = action_policy[:, :-1]
        stop_policy = action_policy[:, -1]
        value = self.value_head(pooled).squeeze(-1)

        return {
            "logits": move_logits,
            "move_logits": move_logits,
            "stop_logits": stop_logits,
            "masked_logits": masked_flat_logits.reshape(batch, n_src, n_tgt),
            "action_logits": action_logits,
            "policy": flat_move_policy.reshape(batch, n_src, n_tgt),
            "move_policy": flat_move_policy.reshape(batch, n_src, n_tgt),
            "stop_policy": stop_policy,
            "action_policy": action_policy,
            "value": value,
        }


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
