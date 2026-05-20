"""Chunked edge policy for experimental Orbit Wars CPU agents.

This model chooses a whole turn plan in one forward pass.  It encodes the
current candidate edge tokens, then decodes ``K`` action slots.  Each slot
points at one attended candidate edge and predicts a continuous ship multiplier
around that edge's harness-computed ship count.

The module is intentionally parallel to ``model.py``.  It is not wired into the
deployed submission path yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness_cpu import FEATURE_DIM, FEATURE_SCALES, GameView_CPU


DEFAULT_D_MODEL = 64
DEFAULT_D_FF = 128
DEFAULT_SLOTS = 10
DEFAULT_ENCODER_LAYERS = 5
DEFAULT_DECODER_LAYERS = 3
MIN_SHIP_MULTIPLIER = 0.25
MAX_SHIP_MULTIPLIER = 3.0


@dataclass
class ChunkedPolicyOutput:
    """Outputs for a batch of chunked turn plans."""

    pointer_logits: torch.Tensor       # (B, K, N + 1), last index is empty
    ship_delta_mu: torch.Tensor        # (B, K), log multiplier mean
    ship_delta_log_std: torch.Tensor   # (B, K), log multiplier std
    active_logits: torch.Tensor        # (B, K)
    value: Optional[torch.Tensor]      # (B,) or None


class FusedQKVAttention(nn.Module):
    """Single-head attention with fused Q/K/V projection."""

    def __init__(self, d_model: int):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        d = x.shape[-1]
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~mask, -1.0e9)
        attn = torch.softmax(scores, dim=-1)
        row_has_any = mask.any(dim=-1, keepdim=True)
        attn = torch.where(row_has_any, attn, torch.zeros_like(attn))
        return self.out(torch.matmul(attn, v))


class CrossAttention(nn.Module):
    """Single-head query-to-context attention."""

    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q(query)
        k = self.k(context)
        v = self.v(context)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~context_mask.unsqueeze(1), -1.0e9)
        attn = torch.softmax(scores, dim=-1)
        row_has_any = context_mask.any(dim=-1, keepdim=True).unsqueeze(1)
        attn = torch.where(row_has_any, attn, torch.zeros_like(attn))
        return self.out(torch.matmul(attn, v))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class AxialEncoderBlock(nn.Module):
    """Two-axis edge attention.

    Odd/even layers can reverse the axis order so neither source nor target
    attention always gets the last update before the feed-forward block.
    """

    def __init__(self, d_model: int, d_ff: int, reverse_order: bool = False):
        super().__init__()
        self.reverse_order = bool(reverse_order)
        self.src_ln = nn.LayerNorm(d_model)
        self.src_attn = FusedQKVAttention(d_model)
        self.tgt_ln = nn.LayerNorm(d_model)
        self.tgt_attn = FusedQKVAttention(d_model)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.reverse_order:
            x = x + self.tgt_attn(self.tgt_ln(x), tgt_mask)
            x = x + self.src_attn(self.src_ln(x), src_mask)
        else:
            x = x + self.src_attn(self.src_ln(x), src_mask)
            x = x + self.tgt_attn(self.tgt_ln(x), tgt_mask)
        x = x + self.ffn(self.ffn_ln(x))
        return x


class SlotDecoderBlock(nn.Module):
    """Slot self-attention plus cross-attention to encoded edge candidates."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.self_ln = nn.LayerNorm(d_model)
        self.self_attn = FusedQKVAttention(d_model)
        self.cross_ln = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        slots: torch.Tensor,
        encoded_edges: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, K, _ = slots.shape
        slot_mask = torch.ones(B, K, K, dtype=torch.bool, device=slots.device)
        slots = slots + self.self_attn(self.self_ln(slots), slot_mask)
        slots = slots + self.cross_attn(self.cross_ln(slots), encoded_edges, valid_mask)
        slots = slots + self.ffn(self.ffn_ln(slots))
        return slots


class ChunkedEdgePolicy(nn.Module):
    """Candidate-grounded action chunk policy.

    The pointer head returns logits over the current candidate token list plus
    one empty class.  Ship control is represented as a log multiplier around
    the selected token's harness-computed ship count.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = DEFAULT_D_MODEL,
        d_ff: int = DEFAULT_D_FF,
        n_slots: int = DEFAULT_SLOTS,
        encoder_layers: int = DEFAULT_ENCODER_LAYERS,
        decoder_layers: int = DEFAULT_DECODER_LAYERS,
        value_hidden: int = 128,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)

        scales = torch.as_tensor(FEATURE_SCALES, dtype=torch.float32)
        if scales.shape[0] != self.feature_dim:
            raise ValueError(
                f"FEATURE_SCALES length {scales.shape[0]} != feature_dim {feature_dim}"
            )
        self.register_buffer("feature_scales", scales)

        self.input_proj = nn.Linear(self.feature_dim, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        self.encoder = nn.ModuleList(
            AxialEncoderBlock(d_model, d_ff, reverse_order=(i % 2 == 1))
            for i in range(encoder_layers)
        )

        self.slot_queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.decoder = nn.ModuleList(
            SlotDecoderBlock(d_model, d_ff) for _ in range(decoder_layers)
        )

        self.pointer_edge = nn.Linear(d_model, d_model, bias=False)
        self.pointer_slot = nn.Linear(d_model, d_model, bias=False)
        self.empty_head = nn.Linear(d_model, 1)
        self.ship_delta = nn.Linear(d_model, 2)
        self.active_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, value_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(value_hidden, value_hidden // 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(value_hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _axial_masks(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_pair = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        src_mask = (src_ids.unsqueeze(2) == src_ids.unsqueeze(1)) & valid_pair
        tgt_mask = (tgt_ids.unsqueeze(2) == tgt_ids.unsqueeze(1)) & valid_pair
        return src_mask, tgt_mask

    def encode_candidates(
        self,
        edges: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = edges / self.feature_scales
        x = self.input_ln(self.input_proj(x))
        src_mask, tgt_mask = self._axial_masks(src_ids, tgt_ids, valid_mask)
        for block in self.encoder:
            x = block(x, src_mask, tgt_mask)
        return x

    def forward(
        self,
        edges: torch.Tensor,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        compute_value: bool = True,
    ) -> ChunkedPolicyOutput:
        B, N, F_in = edges.shape
        if F_in != self.feature_dim:
            raise ValueError(
                f"edges has feature_dim {F_in}, expected {self.feature_dim}"
            )
        if valid_mask is None:
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=edges.device)

        encoded = self.encode_candidates(edges, src_ids, tgt_ids, valid_mask)
        slots = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        for block in self.decoder:
            slots = block(slots, encoded, valid_mask)

        slot_q = self.pointer_slot(slots)
        edge_k = self.pointer_edge(encoded)
        pointer = torch.matmul(slot_q, edge_k.transpose(-1, -2)) / math.sqrt(self.d_model)
        pointer = pointer.masked_fill(~valid_mask.unsqueeze(1), -1.0e9)
        empty = self.empty_head(slots)
        pointer_logits = torch.cat([pointer, empty], dim=-1)

        ship = self.ship_delta(slots)
        ship_delta_mu = ship[..., 0]
        ship_delta_log_std = ship[..., 1].clamp(min=-5.0, max=2.0)
        active_logits = self.active_head(slots).squeeze(-1)

        value = None
        if compute_value:
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
            pooled_edges = (encoded * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
            pooled_slots = slots.mean(dim=1)
            value = self.value_head(torch.cat([pooled_edges, pooled_slots], dim=-1)).squeeze(-1)

        return ChunkedPolicyOutput(
            pointer_logits=pointer_logits,
            ship_delta_mu=ship_delta_mu,
            ship_delta_log_std=ship_delta_log_std,
            active_logits=active_logits,
            value=value,
        )


def multiplier_from_delta(delta: float) -> float:
    """Convert a predicted log multiplier to a bounded ship multiplier."""
    multiplier = math.exp(float(delta))
    return max(MIN_SHIP_MULTIPLIER, min(MAX_SHIP_MULTIPLIER, multiplier))


def decode_chunk_actions(
    view: GameView_CPU,
    token_indices: Sequence[int],
    ship_multipliers: Sequence[float],
    active: Optional[Sequence[bool]] = None,
) -> list[list]:
    """Decode chunk slots into legal Orbit Wars moves.

    This helper is candidate-grounded: each slot references an existing token in
    ``view.tokens()``.  It recomputes the lead angle for the chosen ship count
    and drops slots that fail radar validation or source-budget checks.
    """
    bundle = view.tokens()
    if active is None:
        active = [True] * len(token_indices)
    if len(token_indices) != len(ship_multipliers) or len(token_indices) != len(active):
        raise ValueError("token_indices, ship_multipliers, and active must align")

    remaining_by_src = {
        int(planet[0]): int(planet[5])
        for planet in view.planets
        if int(planet[1]) == int(view.player)
    }
    moves: list[list] = []
    radar = view._get_radar()

    for token_idx, multiplier, is_active in zip(token_indices, ship_multipliers, active):
        if not is_active:
            continue
        token_idx = int(token_idx)
        if not (0 <= token_idx < bundle.n):
            continue

        src_slot = int(bundle.src_ids[token_idx])
        tgt_slot = int(bundle.tgt_ids[token_idx])
        src_pid = int(bundle.planet_ids[src_slot])
        tgt_pid = int(bundle.planet_ids[tgt_slot])
        src = view.planets_by_id.get(src_pid)
        if src is None:
            continue
        remaining = int(remaining_by_src.get(src_pid, 0))
        if remaining <= 0:
            continue

        base_ships = max(1, int(bundle.ships[token_idx]))
        ships = int(round(base_ships * float(multiplier)))
        ships = max(1, min(remaining, ships))

        intercept = view._lead_intercept(
            (float(src[2]), float(src[3])),
            tgt_pid,
            ships,
            src_radius=float(src[4]),
        )
        if intercept is None or not view._sun_crossing_clear((float(src[2]), float(src[3])), intercept):
            continue
        angle = float(intercept["angle"])
        hit = radar.simulate_launch(src, angle, ships)
        if not (hit.hit_planet and int(hit.target_id) == tgt_pid):
            continue

        remaining_by_src[src_pid] = remaining - ships
        moves.append([src_pid, angle, ships])

    return moves
