# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``key_padding_mask`` kwarg added to ``VanillaAttention.forward``.

Exercises the canonical ``[B, S_kv]`` bool mask contract, internal expansion to
``[B, 1, 1, S_kv]``, and the ``is_causal + key_padding_mask`` combination that
SDPA rejects natively (the backend builds an explicit causal AND mask).
"""

import math

import pytest
import torch
import torch.nn.functional as F

try:
    from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
    from tensorrt_llm._torch.visual_gen.attention_backend import VanillaAttention

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not MODULES_AVAILABLE, reason="Required modules not available"
)


def _reference_sdpa(q, k, v, scale, attn_mask=None, is_causal=False):
    """Reference SDPA call used as oracle; matches VanillaAttention's scale."""
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )


class TestVanillaPadMask:
    """Pad-mask behavior for :class:`VanillaAttention.forward`."""

    def test_pad_mask_masks_k(self):
        """VANILLA forward with ``key_padding_mask`` matches SDPA with ``[B, 1, 1, S_kv]`` mask."""
        batch, num_heads, s_q, s_kv, head_dim = 2, 4, 7, 11, 32
        scale = 1.0 / math.sqrt(head_dim)

        torch.manual_seed(0)
        q = torch.randn(batch, num_heads, s_q, head_dim)
        k = torch.randn(batch, num_heads, s_kv, head_dim)
        v = torch.randn(batch, num_heads, s_kv, head_dim)

        # Last 3 K positions are padded on batch 0; last 1 on batch 1.
        pad_mask = torch.ones(batch, s_kv, dtype=torch.bool)
        pad_mask[0, -3:] = False
        pad_mask[1, -1:] = False

        attn = VanillaAttention(num_heads=num_heads, head_dim=head_dim)
        out = attn.forward(q, k, v, key_padding_mask=pad_mask)

        expected = _reference_sdpa(
            q, k, v, scale=scale, attn_mask=pad_mask.view(batch, 1, 1, s_kv)
        )
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_pad_mask_without_mask_matches_full_sdpa(self):
        """Passing ``key_padding_mask=None`` is equivalent to a plain SDPA call (unchanged baseline)."""
        batch, num_heads, s_q, s_kv, head_dim = 2, 4, 5, 7, 32
        scale = 1.0 / math.sqrt(head_dim)

        torch.manual_seed(1)
        q = torch.randn(batch, num_heads, s_q, head_dim)
        k = torch.randn(batch, num_heads, s_kv, head_dim)
        v = torch.randn(batch, num_heads, s_kv, head_dim)

        attn = VanillaAttention(num_heads=num_heads, head_dim=head_dim)
        out = attn.forward(q, k, v)  # No key_padding_mask.

        expected = _reference_sdpa(q, k, v, scale=scale)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_pad_mask_combines_with_causal(self):
        """``attention_mask=CAUSAL`` + ``key_padding_mask`` matches explicit ``causal AND mask`` SDPA."""
        batch, num_heads, seq_len, head_dim = 2, 4, 8, 32
        scale = 1.0 / math.sqrt(head_dim)

        torch.manual_seed(2)
        # Self-attention shape (S_q == S_kv) so the causal mask has a well-defined meaning.
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)

        pad_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        pad_mask[:, -2:] = False  # Last 2 tokens padded across the batch.

        attn = VanillaAttention(num_heads=num_heads, head_dim=head_dim)
        out = attn.forward(
            q,
            k,
            v,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            key_padding_mask=pad_mask,
        )

        # Reference: build causal AND pad mask explicitly, call SDPA with is_causal=False.
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        combined = pad_mask.view(batch, 1, 1, seq_len) & causal
        expected = _reference_sdpa(q, k, v, scale=scale, attn_mask=combined)

        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_pad_mask_drifts_when_omitted(self):
        """Negative test: omitting ``key_padding_mask`` on padded inputs drifts beyond tolerance on valid rows."""
        batch, num_heads, s_q, s_kv, head_dim = 1, 2, 4, 6, 16
        scale = 1.0 / math.sqrt(head_dim)

        torch.manual_seed(3)
        q = torch.randn(batch, num_heads, s_q, head_dim)
        k = torch.randn(batch, num_heads, s_kv, head_dim)
        # Make the padded rows of V obviously large so the unmasked result is polluted.
        v = torch.randn(batch, num_heads, s_kv, head_dim)
        v[:, :, -2:, :] = 100.0

        pad_mask = torch.ones(batch, s_kv, dtype=torch.bool)
        pad_mask[:, -2:] = False

        attn = VanillaAttention(num_heads=num_heads, head_dim=head_dim)
        out_masked = attn.forward(q, k, v, key_padding_mask=pad_mask)
        out_unmasked = attn.forward(q, k, v)

        # Unmasked result is polluted; at least one row diverges well beyond tolerance.
        max_diff = (out_unmasked - out_masked).abs().max().item()
        assert max_diff > 1.0, (
            f"Expected large drift when pad mask is omitted; got max diff {max_diff}"
        )

        # Masked result still equals the SDPA oracle.
        expected = _reference_sdpa(
            q, k, v, scale=scale, attn_mask=pad_mask.view(batch, 1, 1, s_kv)
        )
        torch.testing.assert_close(out_masked, expected, rtol=1e-5, atol=1e-5)

    def test_pad_mask_shape_validation(self):
        """Wrong-rank pad mask raises a clear assertion rather than silently broadcasting."""
        batch, num_heads, s_q, s_kv, head_dim = 2, 2, 4, 6, 16

        q = torch.randn(batch, num_heads, s_q, head_dim)
        k = torch.randn(batch, num_heads, s_kv, head_dim)
        v = torch.randn(batch, num_heads, s_kv, head_dim)

        attn = VanillaAttention(num_heads=num_heads, head_dim=head_dim)

        # Wrong shape: [B, 1, 1, S_kv] — callers must pass [B, S_kv].
        bad_mask = torch.ones(batch, 1, 1, s_kv, dtype=torch.bool)
        with pytest.raises(AssertionError, match="key_padding_mask"):
            attn.forward(q, k, v, key_padding_mask=bad_mask)

        # Wrong dtype: float instead of bool.
        float_mask = torch.ones(batch, s_kv)
        with pytest.raises(AssertionError, match="key_padding_mask"):
            attn.forward(q, k, v, key_padding_mask=float_mask)
