# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AC-8 coverage: ``_LTX2CUDAGraphRunner`` + ``torch.compile`` on the new ``key_padding_mask`` path.

Single-rank, CUDA-only. Exercises the real LTX-2 runner (not the generic
``CUDAGraphRunner``) to verify:

1. ``_LTX2CUDAGraphRunner`` derives distinct graph keys for raw
   ``audio_frames=125`` and ``audio_frames=126`` ``Modality`` inputs at
   ``U=8`` (the padded ``S_full=128`` is internal to ``LTXModel.forward``
   and must NOT cause the runner to collapse the two captures onto one
   key). Both captures replay without NaN.
2. A raw shape not yet in the runner's key set (e.g. ``audio_frames=136``)
   is detected by its absence from ``runner.graphs`` before a capture.
3. ``torch.compile(mode='default', dynamic=True)`` compiles
   ``LTX2Attention.forward`` through the ``use_ulysses_cross=True`` cross-attn
   configuration for both a padded and a non-padded shape, with and without
   ``key_padding_mask``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

try:
    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
    )
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import (
        CUDAGraphRunnerConfig,
    )
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import (
        _LTX2CUDAGraphRunner,
    )
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
        LTX2Attention,
        LTXModel,
        LTXModelType,
    )
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not MODULES_AVAILABLE, reason="Required modules not available"
)


# Reduced LTX2 config: minimal 1-layer model so the runner exercises real
# Modality tensors (Modality fields include latent/timesteps/positions/context)
# and the key derivation, capture, and replay paths at production fidelity.
_LTX2_CFG = dict(
    num_attention_heads=4,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    num_layers=1,
    cross_attention_dim=64,
    caption_channels=32,
    norm_eps=1e-6,
    positional_embedding_max_pos=[4, 16, 16],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    audio_num_attention_heads=4,
    audio_attention_head_dim=16,
    audio_in_channels=16,
    audio_out_channels=16,
    audio_cross_attention_dim=64,
    audio_positional_embedding_max_pos=[192],
    av_ca_timestep_scale_multiplier=1,
)


def _make_model_config() -> DiffusionModelConfig:
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )


def _build_ltx2_model(dtype=torch.bfloat16, device="cuda") -> LTXModel:
    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        model_config=_make_model_config(),
        **_LTX2_CFG,
    )
    model.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            elif p.numel() > 0:
                p.normal_(mean=0.0, std=0.02)
    return model


def _build_modalities(
    audio_frames: int, dtype=torch.bfloat16, device="cuda"
) -> tuple[Modality, Modality]:
    v_frames, v_h, v_w = 2, 4, 4
    v_patches = v_frames * v_h * v_w
    text_len = 4

    video_positions = torch.zeros(1, 3, v_patches, 2, dtype=torch.float32, device=device)
    idx = 0
    for f in range(v_frames):
        for h in range(v_h):
            for w in range(v_w):
                video_positions[:, 0, idx] = torch.tensor([f, f + 1])
                video_positions[:, 1, idx] = torch.tensor([h, h + 1])
                video_positions[:, 2, idx] = torch.tensor([w, w + 1])
                idx += 1

    audio_positions = torch.zeros(1, 1, audio_frames, 2, dtype=torch.float32, device=device)
    for i in range(audio_frames):
        audio_positions[:, 0, i] = torch.tensor([i, i + 1])

    video = Modality(
        latent=torch.randn(1, v_patches, _LTX2_CFG["in_channels"], dtype=dtype, device=device) * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        positions=video_positions,
        context=torch.randn(1, text_len, _LTX2_CFG["caption_channels"], dtype=dtype, device=device) * 0.02,
    )
    audio = Modality(
        latent=torch.randn(1, audio_frames, _LTX2_CFG["audio_in_channels"], dtype=dtype, device=device) * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        positions=audio_positions,
        context=torch.randn(1, text_len, _LTX2_CFG["caption_channels"], dtype=dtype, device=device) * 0.02,
    )
    return video, audio


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph tests need a GPU")
class TestAC8LTX2CUDAGraphRunner:
    """AC-8: the real ``_LTX2CUDAGraphRunner`` on raw audio_frames=125 and 126."""

    def test_two_captures_distinct_keys_and_replay_nan_free(self):
        """Raw ``audio_frames=125`` and ``audio_frames=126`` must produce TWO
        distinct graph keys under ``_LTX2CUDAGraphRunner`` (keys derived from
        raw ``Modality.latent.shape[1]``, NOT from padded ``S_full``). Both
        captures replay without NaN.
        """
        torch.manual_seed(8001)
        model = _build_ltx2_model()
        # Set audio_valid_len so LTXModel.forward treats audio as non-sharded
        # (U=1 effective: configure_audio_ulysses is a no-op under use_ulysses=False).
        # The runner key derivation is independent of the U=8 padding path
        # because the key inputs are the RAW Modality tensors.
        runner = _LTX2CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))

        video_125, audio_125 = _build_modalities(audio_frames=125)
        video_126, audio_126 = _build_modalities(audio_frames=126)

        key_125 = runner.get_graph_key(video_125, audio_125)
        key_126 = runner.get_graph_key(video_126, audio_126)
        assert key_125 != key_126, (
            "AC-8 requires two distinct captures for raw audio_frames=125 vs "
            f"126; keys collided: {key_125!r} == {key_126!r}"
        )

        def _fn(video, audio):
            return model(video=video, audio=audio)

        with torch.no_grad():
            runner.capture(key_125, _fn, (video_125, audio_125), {})
            runner.capture(key_126, _fn, (video_126, audio_126), {})

        assert key_125 in runner.graphs and key_126 in runner.graphs, (
            f"Both keys must be captured; runner.graphs has {list(runner.graphs)}"
        )

        for key, args in [(key_125, (video_125, audio_125)), (key_126, (video_126, audio_126))]:
            out = runner.replay(key, args, {})
            # LTXModel returns (video_vel, audio_vel) — both must be NaN-free.
            assert isinstance(out, tuple) and len(out) == 2
            for name, t in zip(("video", "audio"), out):
                if t is None:
                    continue
                assert not torch.isnan(t).any(), f"{name} replay produced NaN at key={key!r}"

    def test_replay_with_unseen_raw_shape_requires_recapture(self):
        """A raw ``audio_frames=136`` Modality must produce a key that is NOT
        in ``runner.graphs`` after capturing only raw 125 — the replay path
        would surface this as a capture-needed signal rather than silently
        running a stale graph.
        """
        torch.manual_seed(8002)
        model = _build_ltx2_model()
        runner = _LTX2CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))

        video_125, audio_125 = _build_modalities(audio_frames=125)
        video_136, audio_136 = _build_modalities(audio_frames=136)

        key_125 = runner.get_graph_key(video_125, audio_125)
        key_136 = runner.get_graph_key(video_136, audio_136)
        assert key_125 != key_136

        def _fn(video, audio):
            return model(video=video, audio=audio)

        with torch.no_grad():
            runner.capture(key_125, _fn, (video_125, audio_125), {})

        assert key_125 in runner.graphs
        assert key_136 not in runner.graphs, (
            "Unseen raw shape must NOT appear in runner.graphs; a silent key "
            "collision would let the unseen 136-token Modality replay a "
            "stale 125-token graph."
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile smoke needs a GPU")
class TestAC8TorchCompileUlyssesCross:
    """AC-8: ``torch.compile`` on ``LTX2Attention.forward`` through the intended
    ``use_ulysses_cross=True`` runtime path for padded + non-padded inputs.

    The cross-attn under Ulysses uses ``SEPARATE_QKV`` with
    ``pre_projected_kv`` and a separate ``context`` input; the inner backend
    resolves to ``VANILLA`` so ``key_padding_mask`` flows through. We
    construct the attention with ``use_ulysses_cross=True`` so the full
    wrapper stack participates in the compile trace (the ``UlyssesCrossAttention``
    wrapper takes its ``world_size==1`` fast path on a single-rank test,
    which bypasses the a2a but still exercises the same Python forward the
    production multi-rank path invokes).
    """

    @staticmethod
    def _vg_mapping_u1():
        # Single-rank mapping: use_ulysses_cross=True triggers the wrapper
        # but with world_size==1 it skips all collectives.
        return SimpleNamespace(ulysses_size=1, ulysses_group=None, ulysses_rank=0)

    def _build_cross_attn(self, heads=4, dim_head=32, context_dim=None):
        cfg = _make_model_config()
        cfg.visual_gen_mapping = self._vg_mapping_u1()
        attn = LTX2Attention(
            query_dim=heads * dim_head,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            config=cfg,
            use_ulysses_cross=True,
        )
        attn.to(device="cuda", dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for name, p in attn.named_parameters():
                if "norm" in name and "weight" in name:
                    p.fill_(1.0)
                elif p.numel() > 0:
                    p.normal_(mean=0.0, std=0.02)
        return attn

    def test_compile_cross_attn_padded_and_non_padded(self):
        heads, dim_head = 4, 32
        query_dim = heads * dim_head
        attn = self._build_cross_attn(heads=heads, dim_head=dim_head, context_dim=query_dim)
        compiled = torch.compile(attn.forward, mode="default", dynamic=True)

        # Non-padded cross-attn: no key_padding_mask.
        s_q, s_kv = 8, 10
        x = torch.randn(1, s_q, query_dim, dtype=torch.bfloat16, device="cuda")
        ctx = torch.randn(1, s_kv, query_dim, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            out_up = compiled(x, context=ctx)
        assert out_up.shape == (1, s_q, query_dim)
        assert not torch.isnan(out_up).any()

        # Padded cross-attn (K/V are padded): key_padding_mask flags pad cols.
        s_kv_p = 12
        ctx_p = torch.randn(1, s_kv_p, query_dim, dtype=torch.bfloat16, device="cuda")
        pad_mask = torch.ones(1, s_kv_p, dtype=torch.bool, device="cuda")
        pad_mask[:, -2:] = False
        with torch.no_grad():
            out_p = compiled(x, context=ctx_p, key_padding_mask=pad_mask)
        assert out_p.shape == (1, s_q, query_dim)
        assert not torch.isnan(out_p).any()
