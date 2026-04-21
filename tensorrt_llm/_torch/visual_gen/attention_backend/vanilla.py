# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Diffusion Vanilla Attention Backend

Simple attention implementation for visual generation (diffusion) models using
torch.nn.functional.scaled_dot_product_attention (SDPA).

Supports both self-attention and cross-attention (different Q/KV sequence lengths).
No KV cache - full recompute each diffusion step.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout


class VanillaAttention(AttentionBackend):
    """
    Vanilla Attention for diffusion models using torch SDPA.

    Uses torch.nn.functional.scaled_dot_product_attention which:
    - Properly handles cross-attention (different Q/KV sequence lengths)
    - Uses Flash Attention 2 when available (via SDPA backend selection)
    - No KV cache needed for diffusion models

    This is simpler than the LLM VanillaAttention which has complex
    KV cache handling and uses flash_attn_varlen_func.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.scale = 1.0 / math.sqrt(head_dim)

        # SDPA expects [B, H, S, D] format
        self._preferred_layout = AttentionTensorLayout.HND

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using torch SDPA.

        Dimensions are derived from tensor shapes (HND layout: ``[B, H, S, D]``).

        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_kv_heads, seq_len_kv, head_dim]
            v: Value tensor [batch_size, num_kv_heads, seq_len_kv, head_dim]
            attention_mask: Attention mask type (CAUSAL or FULL)
            key_padding_mask: Optional bool tensor ``[batch_size, seq_len_kv]``
                where ``True`` means the K/V position is valid and ``False``
                means padded. Expanded internally to ``[B, 1, 1, S_kv]`` to
                broadcast across heads and query positions. When combined with
                ``attention_mask=CAUSAL`` the causal mask is built explicitly
                and AND-ed with the pad mask (SDPA rejects ``attn_mask`` +
                ``is_causal=True``).

        Returns:
            Output tensor [batch_size, num_heads, seq_len, head_dim]
        """
        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL

        assert q.dim() == 4 and q.shape[3] == self.head_dim, (
            f"Invalid q shape: expected [B, H, S, D={self.head_dim}], got {q.shape}"
        )
        assert k.dim() == 4 and k.shape[0] == q.shape[0] and k.shape[3] == self.head_dim, (
            f"Invalid k shape: expected [B={q.shape[0]}, H_kv, S_kv, D={self.head_dim}], got {k.shape}"
        )
        assert v.dim() == 4 and v.shape[0] == q.shape[0] and v.shape[3] == self.head_dim, (
            f"Invalid v shape: expected [B={q.shape[0]}, H_kv, S_kv, D={self.head_dim}], got {v.shape}"
        )

        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            B, S_kv = q.shape[0], k.shape[2]
            assert key_padding_mask.dtype == torch.bool, (
                f"key_padding_mask must be bool, got {key_padding_mask.dtype}"
            )
            assert key_padding_mask.shape == (B, S_kv), (
                f"key_padding_mask must have shape [B={B}, S_kv={S_kv}], "
                f"got {tuple(key_padding_mask.shape)}"
            )
            # Expand [B, S_kv] -> [B, 1, 1, S_kv] so it broadcasts over heads and Q positions.
            attn_mask = key_padding_mask.view(B, 1, 1, S_kv)
            if is_causal:
                # SDPA rejects attn_mask + is_causal=True; build the causal mask explicitly.
                S_q = q.shape[2]
                causal = torch.ones(
                    S_q, S_kv, dtype=torch.bool, device=q.device
                ).tril()
                attn_mask = attn_mask & causal
                is_causal = False

        sdpa_kwargs = {}
        if self.num_kv_heads != self.num_heads:
            # PyTorch 2.5+: SDPA honors the GQA ratio when enable_gqa=True.
            sdpa_kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale,
            **sdpa_kwargs,
        )

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
