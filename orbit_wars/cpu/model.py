"""Dynamic-edge transformer for Orbit Wars — CPU inference focus.

Operates on packed edge tokens emitted by ``harness_cpu.GameView_CPU``.

Attention pattern is threshold-gated per sample:
    N_sample ≤ axial_threshold  → full self-attention among valid tokens
    N_sample  > axial_threshold → axial: layer 1 attends within same src,
                                         layer 2 attends within same tgt

Single-head attention with a fused Q/K/V projection (one Linear(d, 3d))
keeps CPU BLAS efficient.  Pre-LayerNorm, GELU activations.

Value head uses attention pooling with a learned query and an MLP large
enough to be useful for PPO advantage estimation.  It is gated by
``compute_value`` so Kaggle inference skips it entirely.

Forward:
    logits (B, N+1), value (B,)|None = model(edges, src_ids, tgt_ids,
                                             valid_mask=None,
                                             compute_value=True)

The last slot in logits is the stop action.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness_cpu import FEATURE_DIM, FEATURE_SCALES


DEFAULT_D_MODEL = 32
DEFAULT_D_FF = 64
DEFAULT_LAYERS = 2
DEFAULT_AXIAL_THRESHOLD = 256
DEFAULT_VALUE_HIDDEN = 128


class FusedQKVAttention(nn.Module):
    """Single-head self-attention with fused Q/K/V projection.

    mask: (B, N, N) bool — True means "attention allowed."
    Rows with no allowed keys get a zero output (not NaN).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)                                          # (B, N, 3d)
        q, k, v = qkv.chunk(3, dim=-1)                             # (B, N, d) each
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, N, N)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        # Rows where every key was masked produce uniform NaN-adjacent values
        # after softmax; zero them so pad positions don't leak signal.
        row_has_any = mask.any(dim=-1, keepdim=True)               # (B, N, 1)
        attn = torch.where(row_has_any, attn, torch.zeros_like(attn))
        out = torch.matmul(attn, v)                                # (B, N, d)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class EncoderBlock(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FusedQKVAttention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class AttentionPool(nn.Module):
    """Learned-query attention pooling over (B, N, d) with a (B, N) valid mask.

    Returns (B, d).  Samples whose valid_mask is all False get a zero vector.
    N=0 is handled explicitly (some batches may have zero-token samples).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        if N == 0:
            return torch.zeros(B, d, dtype=x.dtype, device=x.device)
        q = self.query.view(1, 1, d).expand(B, 1, d)                # (B, 1, d)
        scores = torch.matmul(q, x.transpose(-1, -2)) * self.scale  # (B, 1, N)
        scores = scores.masked_fill(~valid_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        row_has_any = valid_mask.any(dim=-1, keepdim=True).unsqueeze(1)  # (B, 1, 1)
        attn = torch.where(row_has_any, attn, torch.zeros_like(attn))
        pooled = torch.matmul(attn, x).squeeze(1)                    # (B, d)
        return pooled


class ValueHead(nn.Module):
    """Attention-pool + 4-layer MLP → scalar.

    Training-only by default (gated via ``compute_value`` in the main model).
    """

    def __init__(self, d_model: int, hidden: int = DEFAULT_VALUE_HIDDEN):
        super().__init__()
        self.pool = AttentionPool(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x, valid_mask)
        return self.mlp(pooled).squeeze(-1)


class OrbitWarsEdgeTransformer(nn.Module):
    """Edge-set transformer for Orbit Wars with threshold-gated attention."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = DEFAULT_D_MODEL,
        d_ff: int = DEFAULT_D_FF,
        n_layers: int = DEFAULT_LAYERS,
        axial_threshold: int = DEFAULT_AXIAL_THRESHOLD,
        value_hidden: int = DEFAULT_VALUE_HIDDEN,
    ):
        super().__init__()
        if n_layers != 2:
            raise ValueError("Model hard-codes 2 layers (src-axial + tgt-axial).")
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.axial_threshold = int(axial_threshold)

        scales = torch.as_tensor(FEATURE_SCALES, dtype=torch.float32)
        if scales.shape[0] != feature_dim:
            raise ValueError(
                f"FEATURE_SCALES length {scales.shape[0]} != feature_dim {feature_dim}"
            )
        self.register_buffer("feature_scales", scales)

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)

        self.block1 = EncoderBlock(d_model, d_ff)
        self.block2 = EncoderBlock(d_model, d_ff)

        self.edge_head = nn.Linear(d_model, 1)
        self.stop_pool = AttentionPool(d_model)
        self.stop_head = nn.Linear(d_model, 1)

        self.value_head = ValueHead(d_model, hidden=value_hidden)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_masks(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """Return (mask_layer1, mask_layer2) of shape (B, N, N) bool.

        If no sample in the batch exceeds axial_threshold, both layers get
        the cheap full-attention mask and we skip building the axial ones.
        """
        B, N = valid_mask.shape
        valid_pair = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)  # (B, N, N)
        n_valid = valid_mask.sum(dim=-1)                                # (B,)
        needs_axial = (n_valid > self.axial_threshold)                  # (B,)

        if not bool(needs_axial.any()):
            return valid_pair, valid_pair

        shares_src = (src_ids.unsqueeze(2) == src_ids.unsqueeze(1))
        shares_tgt = (tgt_ids.unsqueeze(2) == tgt_ids.unsqueeze(1))
        axial1 = shares_src & valid_pair
        axial2 = shares_tgt & valid_pair

        gate = needs_axial.view(B, 1, 1)
        mask1 = torch.where(gate, axial1, valid_pair)
        mask2 = torch.where(gate, axial2, valid_pair)
        return mask1, mask2

    def forward(
        self,
        edges: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        compute_value: bool = True,
    ):
        """
        Args:
            edges:       (B, N, FEATURE_DIM) float32, raw units (scaled inside).
            src_ids:     (B, N) long — slot ids; pad positions can be any value.
            tgt_ids:     (B, N) long — ditto.
            valid_mask:  (B, N) bool; defaults to all-True.
            compute_value: run the value head when True.

        Returns:
            logits: (B, N+1) — index N is the stop action.
            value:  (B,) float, or None if compute_value=False.
        """
        B, N, F_in = edges.shape
        if F_in != self.feature_dim:
            raise ValueError(
                f"edges has feature_dim {F_in}, expected {self.feature_dim}"
            )
        if valid_mask is None:
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=edges.device)

        x = edges / self.feature_scales                  # (B, N, F)
        x = self.input_ln(self.input_proj(x))            # (B, N, d)

        mask1, mask2 = self._build_masks(src_ids, tgt_ids, valid_mask)

        x = self.block1(x, mask1)
        x = self.block2(x, mask2)

        edge_logits = self.edge_head(x).squeeze(-1)      # (B, N)
        edge_logits = edge_logits.masked_fill(~valid_mask, -1e9)

        stop_pooled = self.stop_pool(x, valid_mask)      # (B, d)
        stop_logit = self.stop_head(stop_pooled)         # (B, 1)

        logits = torch.cat([edge_logits, stop_logit], dim=-1)  # (B, N+1)

        value = None
        if compute_value:
            value = self.value_head(x, valid_mask)

        return logits, value


def count_parameters(model: nn.Module, include_value: bool = True) -> int:
    total = 0
    for name, p in model.named_parameters():
        if not include_value and name.startswith("value_head"):
            continue
        total += p.numel()
    return total
