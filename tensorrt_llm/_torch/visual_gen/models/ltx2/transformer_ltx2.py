# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

# Architecture ported from LTX-2,
# with compute-heavy components replaced by TRT-LLM optimized modules:
#   - Linear projections  → tensorrt_llm._torch.modules.linear.Linear
#   - RMSNorm (QK norm)   → tensorrt_llm._torch.modules.rms_norm.RMSNorm
#   - FeedForward (MLP)    → tensorrt_llm._torch.modules.mlp.MLP
#   - Attention backend    → tensorrt_llm._torch.visual_gen.attention_backend
#
# Architecture-specific components (RoPE, AdaLN, timestep/text embeddings,
# modality dataclass, transformer args) are ported from LTX-2 and live
# in the ltx2_core/ subpackage.

# TODO: replace torch rms_norm with TRT-LLM RMSNorm (no weights)

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.visual_gen.attention_backend.parallel import UlyssesCrossAttention
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from .ltx2_core.adaln import AdaLayerNormSingle
from .ltx2_core.modality import Modality
from .ltx2_core.perturbations import BatchedPerturbationConfig, PerturbationType
from .ltx2_core.rope import LTXRopeType, apply_rotary_emb
from .ltx2_core.text_projection import PixArtAlphaTextProjection
from .ltx2_core.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from .ltx2_core.utils_ltx2 import rms_norm

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


# ---------------------------------------------------------------------------
# LTX2Attention: TRT-LLM Linear + RMSNorm + attention backend + LTX-2 RoPE
# ---------------------------------------------------------------------------


class LTX2Attention(Attention):
    """LTX-2 attention: extends base Attention with LTX-specific RoPE, gated
    attention, and separate K-RoPE for audio-video cross-attention.

    Inherits from base Attention:
    - Q/K/V Linear creation with quant_config propagation
    - QK RMSNorm (norm_q / norm_k)
    - Backend dispatch with automatic HND/NHD layout handling (_attn_impl)
    - Output projection (to_out)

    Adds LTX-2 specifics:
    - LTX 3D RoPE (INTERLEAVED / SPLIT) with separate k_pe support
    - Gated attention (to_gate_logits)
    - Cross-attention with different context_dim for K/V input
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
        use_ulysses: bool = False,
        use_ulysses_cross: bool = False,
    ):
        from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

        config = config or DiffusionModelConfig()
        vgm = config.visual_gen_mapping

        # Store before super().__init__() — _init_qkv_proj needs _context_dim
        self._context_dim = context_dim if context_dim is not None else query_dim
        self.rope_type = rope_type
        self._is_cross_attn = context_dim is not None

        # Self-attention: FUSE_QKV enables the optimized backend + auto Ulysses
        # wrapping from the base class.
        # Cross-attention: SEPARATE_QKV since K/V come from a different source.
        qkv_mode = QKVMode.SEPARATE_QKV if self._is_cross_attn else QKVMode.FUSE_QKV

        super().__init__(
            hidden_size=query_dim,
            num_attention_heads=heads,
            head_dim=dim_head,
            qkv_mode=qkv_mode,
            qk_norm=True,
            qk_norm_mode="full",
            eps=norm_eps,
            bias=True,
            config=config,
            layer_idx=layer_idx,
        )

        # For audio self-attention that may need a runtime Ulysses toggle
        # (e.g. Stage 2 of the two-stage pipeline disables Ulysses and runs
        # on a single rank), create a plain backend as fallback. The base
        # class already set self.attn to UlyssesAttention(inner_backend=...).
        self._has_dual_attn = False
        ulysses_size = vgm.ulysses_size if vgm is not None else 1
        if use_ulysses and not self._is_cross_attn and ulysses_size > 1:
            self._ulysses_attn = self.attn
            self._plain_attn = create_attention(
                backend=self.attn_backend,
                layer_idx=self.layer_idx,
                num_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_key_value_heads,
                quant_config=self.quant_config,
                dtype=self.dtype,
            )
            self._has_dual_attn = True

        # Strict-Ulysses cross-attention: wrap the resolved backend with
        # UlyssesCrossAttention (unfused Q a2a + fused K|V 5D a2a + output
        # a2a). Kept as a dual-attn pair so Stage 2 can flip back to the
        # plain inner backend via set_ulysses_active(False). Only applies to
        # cross-attention modules (SEPARATE_QKV, pre-projected K/V).
        self._has_dual_cross_attn = False
        if use_ulysses_cross and self._is_cross_attn and ulysses_size > 1:
            H = self.num_attention_heads
            H_kv = self.num_key_value_heads
            U = ulysses_size
            if H % U != 0 or H_kv % U != 0:
                raise ValueError(
                    "UlyssesCrossAttention requires num_heads and num_kv_heads "
                    f"divisible by ulysses_size; got H={H}, H_kv={H_kv}, U={U}"
                )
            inner_cross = create_attention(
                backend=self.attn_backend,
                layer_idx=self.layer_idx,
                num_heads=H // U,
                num_kv_heads=H_kv // U,
                head_dim=self.head_dim,
                quant_config=self.quant_config,
                dtype=self.dtype,
            )
            self._ulysses_cross_attn = UlyssesCrossAttention(
                inner_backend=inner_cross,
                process_group=vgm.ulysses_group,
            )
            self._plain_cross_attn = self.attn  # Existing non-Ulysses attn.
            self.attn = self._ulysses_cross_attn
            self._has_dual_cross_attn = True

        if apply_gated_attention:
            self.to_gate_logits = Linear(
                query_dim,
                heads,
                bias=True,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )
        else:
            self.to_gate_logits = None

    def set_ulysses_active(self, active: bool):
        """Toggle between Ulysses-wrapped and plain attention at runtime.

        Effective for modules created with ``use_ulysses=True`` (self-attn
        dual-attn pair) and/or ``use_ulysses_cross=True`` (cross-attn
        dual-attn pair). Both pairs flip together so a single call from
        ``TransformerLTX2.set_ulysses_enabled`` covers self- and cross-attn.
        """
        if not (self._has_dual_attn or self._has_dual_cross_attn):
            return
        self._modules.pop("attn", None)
        if self._has_dual_cross_attn:
            self.attn = self._ulysses_cross_attn if active else self._plain_cross_attn
        else:
            self.attn = self._ulysses_attn if active else self._plain_attn

    def _init_qkv_proj(self):
        """Override for cross-attention: use _context_dim for K/V input.

        Self-attention delegates to the base class which creates a fused
        qkv_proj (FUSE_QKV).
        """
        if not self._is_cross_attn:
            super()._init_qkv_proj()
            return
        self.to_q = Linear(
            self.hidden_size,
            self.q_dim,
            bias=self.bias,
            dtype=self.dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )
        self.to_k = Linear(
            self._context_dim,
            self.kv_dim,
            bias=self.bias,
            dtype=self.dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )
        self.to_v = Linear(
            self._context_dim,
            self.kv_dim,
            bias=self.bias,
            dtype=self.dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )

    def project_kv(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and normalize K/V from context.

        Used by the project-before-gather pattern in AV cross-attention:
        project K/V on sharded data, then all-gather the smaller projected
        tensors instead of all-gathering the full context first.
        """
        k = self.to_k(context)
        v = self.to_v(context)
        if self.qk_norm:
            k = self.norm_k(k)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        pre_projected_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query input [B, T, D].
            context: Key/value input [B, S, C]. None → self-attention.
            pe: (cos, sin) RoPE embeddings for Q (and K when k_pe is None).
            k_pe: Separate (cos, sin) RoPE embeddings for K (for AV cross-attn).
            pre_projected_kv: Pre-projected (k, v) tuple from project_kv().
                When provided, skips K/V projection and K-norm (already done).
            key_padding_mask: Optional ``[B, S_kv]`` bool tensor (True = valid,
                False = pad). Full-seq, identical across Ulysses ranks. Passed
                through ``_attn_impl`` to the inner attention backend, which
                expands it to ``[B, 1, 1, S_kv]`` for SDPA.
        """
        if pre_projected_kv is not None:
            k, v = pre_projected_kv
            q = self.to_q(x)
            if self.qk_norm:
                q = self.norm_q(q)
        else:
            q, k, v = self.get_qkv(x, context)
            q, k = self.apply_qk_norm(q, k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        attn_kwargs = {}
        if key_padding_mask is not None:
            attn_kwargs["key_padding_mask"] = key_padding_mask
        out = self._attn_impl(q, k, v, **attn_kwargs)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.num_attention_heads, self.head_dim)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.num_attention_heads * self.head_dim)

        return self.to_out[0](out)


# ---------------------------------------------------------------------------
# TransformerConfig + BasicAVTransformerBlock
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False


class BasicAVTransformerBlock(nn.Module):
    """Dual-stream (Audio/Video) transformer block using TRT-LLM primitives.

    Each block contains per-modality self-attention, cross-attention (text),
    bidirectional AV cross-attention, and FFN — all with AdaLN modulation.
    """

    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        config: Optional["DiffusionModelConfig"] = None,
    ):
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps

        self._use_ulysses = False
        self._audio_is_sharded = False
        vgm = config.visual_gen_mapping if config is not None else None
        if vgm is not None and vgm.ulysses_size > 1:
            self._use_ulysses = True
            self._ulysses_size = vgm.ulysses_size
            self._ulysses_pg = vgm.ulysses_group

        if video is not None:
            self._init_video_modules(video, rope_type, norm_eps, config, idx)

        if audio is not None:
            self._init_audio_modules(audio, rope_type, norm_eps, config, idx)

        if audio is not None and video is not None:
            self._init_av_cross_modules(video, audio, rope_type, norm_eps, config, idx)

    @staticmethod
    def _make_mlp(cfg, model_config, idx):
        dtype = model_config.torch_dtype if model_config else None
        return MLP(
            hidden_size=cfg.dim,
            intermediate_size=cfg.dim * 4,
            bias=True,
            activation=lambda x: F.gelu(x, approximate="tanh"),
            dtype=dtype,
            config=model_config,
            layer_idx=idx,
        )

    def _init_video_modules(self, cfg, rope_type, eps, model_config, idx):
        self.attn1 = LTX2Attention(
            query_dim=cfg.dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
            use_ulysses=True,
        )
        self.attn2 = LTX2Attention(
            query_dim=cfg.dim,
            context_dim=cfg.context_dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
        )
        self.ff = self._make_mlp(cfg, model_config, idx)
        self.scale_shift_table = nn.Parameter(torch.empty(6, cfg.dim))

    def _init_audio_modules(self, cfg, rope_type, eps, model_config, idx):
        self.audio_attn1 = LTX2Attention(
            query_dim=cfg.dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
            use_ulysses=True,
        )
        # Under Ulysses, audio self-attention consumes a key_padding_mask for
        # padded audio tokens. Only the VANILLA backend honors that kwarg
        # today (flash_attn4.py and trtllm.py ignore it), so a non-VANILLA
        # resolved inner backend under Ulysses would silently drop padding
        # correctness. Hard-error at construction with an actionable message.
        self._assert_resolved_vanilla(
            "audio_attn1", self.audio_attn1, model_config
        )
        self.audio_attn2 = LTX2Attention(
            query_dim=cfg.dim,
            context_dim=cfg.context_dim,
            heads=cfg.heads,
            dim_head=cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
        )
        self.audio_ff = self._make_mlp(cfg, model_config, idx)
        self.audio_scale_shift_table = nn.Parameter(torch.empty(6, cfg.dim))

    def _init_av_cross_modules(self, v_cfg, a_cfg, rope_type, eps, model_config, idx):
        self.audio_to_video_attn = LTX2Attention(
            query_dim=v_cfg.dim,
            context_dim=a_cfg.dim,
            heads=a_cfg.heads,
            dim_head=a_cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=v_cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
            use_ulysses_cross=True,
        )
        # a2v consumes key_padding_mask for padded audio K/V; require VANILLA
        # inner backend under Ulysses (AC-7.3). The check inspects the
        # RESOLVED backend, so a user-configured TRTLLM that falls back to
        # VANILLA via the SEPARATE_QKV substitution in modules/attention.py
        # passes cleanly.
        self._assert_resolved_vanilla(
            "audio_to_video_attn", self.audio_to_video_attn, model_config
        )
        self.video_to_audio_attn = LTX2Attention(
            query_dim=a_cfg.dim,
            context_dim=v_cfg.dim,
            heads=a_cfg.heads,
            dim_head=a_cfg.d_head,
            rope_type=rope_type,
            norm_eps=eps,
            apply_gated_attention=a_cfg.apply_gated_attention,
            config=model_config,
            layer_idx=idx,
            use_ulysses_cross=True,
        )
        # v2a does NOT receive key_padding_mask (video K/V is unpadded and
        # padded Q rows are stripped on exit by LTXModel.forward), so any
        # resolved backend is acceptable — no assertion here.
        self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, a_cfg.dim))
        self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, v_cfg.dim))

    def _assert_rank_consistent_flags(
        self,
        *,
        run_a2v: bool,
        run_v2a: bool,
        skip_a2v: bool,
        skip_v2a: bool,
    ) -> None:
        """Env-gated debug check: branch flags must be identical across the Ulysses group.

        Broadcasts each flag from rank 0 and asserts the local value matches.
        Adds four extra NCCL collectives per block per step — enabled only
        via ``VG_DEBUG_RANK_CONSISTENCY=1`` for bring-up / regression hunts.
        """
        device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        flags = torch.tensor(
            [int(run_a2v), int(run_v2a), int(skip_a2v), int(skip_v2a)],
            dtype=torch.int32,
            device=device,
        )
        expected = flags.clone()
        dist.broadcast(expected, src=0, group=self._ulysses_pg)
        if not torch.equal(flags, expected):
            local = flags.tolist()
            src = expected.tolist()
            names = ("run_a2v", "run_v2a", "skip_a2v", "skip_v2a")
            diverged = [
                f"{names[i]}: local={bool(local[i])}, rank0={bool(src[i])}"
                for i in range(4)
                if local[i] != src[i]
            ]
            raise AssertionError(
                "BasicAVTransformerBlock.forward branch flags diverged across "
                f"the Ulysses group at block idx={self.idx}: " + "; ".join(diverged)
            )

    @staticmethod
    def _assert_resolved_vanilla(name: str, attn: "LTX2Attention", model_config) -> None:
        """Hard-error when ``attn`` resolves to a non-VANILLA backend under Ulysses.

        Applied to ``audio_attn1`` (self-attn with padded audio) and
        ``audio_to_video_attn`` (video-Q attends padded-audio K/V). Both
        consume ``key_padding_mask`` which only ``VanillaAttention`` honors.

        The check runs against the RESOLVED backend (``attn.attn_backend``)
        after the TRTLLM → VANILLA substitution inside
        ``modules/attention.py``: a configured TRTLLM for SEPARATE_QKV
        already resolves to VANILLA and passes.
        """
        vgm = getattr(model_config, "visual_gen_mapping", None) if model_config else None
        ulysses_size = vgm.ulysses_size if vgm is not None else 1
        if ulysses_size <= 1:
            return
        resolved = getattr(attn, "attn_backend", None)
        if resolved != "VANILLA":
            raise ValueError(
                f"{name} requires the VANILLA attention backend under Ulysses "
                f"(ulysses_size={ulysses_size}) because it consumes "
                f"key_padding_mask, which other backends silently drop. "
                f"Resolved backend is {resolved!r}. Set "
                "DiffusionModelConfig.attention.backend='VANILLA' or choose "
                "a config where the backend resolves to VANILLA via the "
                "SEPARATE_QKV fallback in modules/attention.py."
            )

    # -- AdaLN helpers -------------------------------------------------------

    @staticmethod
    def _get_ada_values(
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        ada_values = (
            scale_shift_table[indices]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    @staticmethod
    def _get_av_ca_ada_values(
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        ss_table = scale_shift_table[:num_scale_shift_values, :]
        gate_table = scale_shift_table[num_scale_shift_values:, :]

        ss_vals = (
            ss_table.unsqueeze(0)
            .unsqueeze(0)
            .to(device=scale_shift_timestep.device, dtype=scale_shift_timestep.dtype)
            + scale_shift_timestep.reshape(
                batch_size, scale_shift_timestep.shape[1], num_scale_shift_values, -1
            )
        ).unbind(dim=2)

        gate_vals = (
            gate_table.unsqueeze(0)
            .unsqueeze(0)
            .to(device=gate_timestep.device, dtype=gate_timestep.dtype)
            + gate_timestep.reshape(
                batch_size, gate_timestep.shape[1], num_ada_params - num_scale_shift_values, -1
            )
        ).unbind(dim=2)

        ss_chunks = [t.squeeze(2) for t in ss_vals]
        gate_chunks = [t.squeeze(2) for t in gate_vals]
        return (*ss_chunks, *gate_chunks)

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations=None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Forward with optional perturbation masking for STG.

        Rank-invariance invariant: when Ulysses is active (``self._use_ulysses
        and self._ulysses_size > 1``), the branch flags ``run_a2v``,
        ``run_v2a``, ``skip_a2v``, ``skip_v2a`` MUST be identical on every
        rank in ``self._ulysses_pg``. They gate shared collectives inside the
        ``audio_to_video_attn`` / ``video_to_audio_attn`` wrappers; divergence
        would deadlock the all-to-all. Setting
        ``VG_DEBUG_RANK_CONSISTENCY=1`` enables a broadcast-based check below.

        Args:
            perturbations: Optional ``BatchedPerturbationConfig`` that masks
                attention outputs for selected blocks/modalities.
        """
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        has_perturbations = perturbations is not None and isinstance(
            perturbations, BatchedPerturbationConfig
        )

        # --- Video self-attention + text cross-attention ---
        if run_vx:
            skip_v_self = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx
            )
            vshift_msa, vscale_msa, vgate_msa = self._get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            if not skip_v_self:
                with nvtx_range("ltx2.video_self_attn"):
                    norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
                    v_self_out = self.attn1(norm_vx, pe=video.positional_embeddings) * vgate_msa
                    if has_perturbations and perturbations.any_in_batch(
                        PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx
                    ):
                        v_self_out = v_self_out * perturbations.mask_like(
                            PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, v_self_out
                        )
                    vx = vx + v_self_out
            with nvtx_range("ltx2.t2v_cross_attn"):
                vx = vx + self.attn2(
                    rms_norm(vx, eps=self.norm_eps),
                    context=video.context,
                )
            del vshift_msa, vscale_msa, vgate_msa

        # --- Audio self-attention + text cross-attention ---
        if run_ax:
            skip_a_self = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx
            )
            ashift_msa, ascale_msa, agate_msa = self._get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            if not skip_a_self:
                with nvtx_range("ltx2.audio_self_attn"):
                    norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
                    a_self_out = (
                        self.audio_attn1(
                            norm_ax,
                            pe=audio.positional_embeddings,
                            key_padding_mask=audio.audio_padding_mask,
                        )
                        * agate_msa
                    )
                    if has_perturbations and perturbations.any_in_batch(
                        PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx
                    ):
                        a_self_out = a_self_out * perturbations.mask_like(
                            PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, a_self_out
                        )
                    ax = ax + a_self_out
            with nvtx_range("ltx2.t2a_cross_attn"):
                ax = ax + self.audio_attn2(
                    rms_norm(ax, eps=self.norm_eps),
                    context=audio.context,
                )
            del ashift_msa, ascale_msa, agate_msa

        # --- Bidirectional audio ↔ video cross-attention ---
        if run_a2v or run_v2a:
            skip_a2v = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx
            )
            skip_v2a = has_perturbations and perturbations.all_in_batch(
                PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx
            )

            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            (
                scale_ca_audio_a2v,
                shift_ca_audio_a2v,
                scale_ca_audio_v2a,
                shift_ca_audio_v2a,
                gate_out_v2a,
            ) = self._get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_a2v,
                shift_ca_video_a2v,
                scale_ca_video_v2a,
                shift_ca_video_v2a,
                gate_out_a2v,
            ) = self._get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            # Env-gated rank-consistency check (off by default — zero
            # production cost). The AV cross-attn wrappers launch shared
            # collectives gated on the flags above; a divergent flag would
            # deadlock. When VG_DEBUG_RANK_CONSISTENCY=1 and Ulysses is
            # active, broadcast each flag from rank 0 and assert. Stage 2
            # (where self._use_ulysses is False) is a no-op.
            if (
                self._use_ulysses
                and getattr(self, "_ulysses_size", 1) > 1
                and os.environ.get("VG_DEBUG_RANK_CONSISTENCY") == "1"
            ):
                self._assert_rank_consistent_flags(
                    run_a2v=run_a2v,
                    run_v2a=run_v2a,
                    skip_a2v=skip_a2v,
                    skip_v2a=skip_v2a,
                )

            if run_a2v and not skip_a2v:
                with nvtx_range("ltx2.a2v_cross_attn"):
                    vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                    ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v

                    # Strict-Ulysses A2A pattern: UlyssesCrossAttention runs
                    # Q a2a + fused K|V 5D a2a + output a2a internally. No
                    # pre-gather helpers; K/V are passed sharded on their own
                    # seq axis and the wrapper handles distribution.
                    k_a2v, v_a2v = self.audio_to_video_attn.project_kv(ax_scaled)

                    a2v_out = (
                        self.audio_to_video_attn(
                            vx_scaled,
                            pre_projected_kv=(k_a2v, v_a2v),
                            pe=video.cross_positional_embeddings,
                            k_pe=audio.cross_positional_embeddings,
                            key_padding_mask=audio.audio_padding_mask,
                        )
                        * gate_out_a2v
                    )
                    if has_perturbations and perturbations.any_in_batch(
                        PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx
                    ):
                        a2v_out = a2v_out * perturbations.mask_like(
                            PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, a2v_out
                        )
                    vx = vx + a2v_out

            if run_v2a and not skip_v2a:
                with nvtx_range("ltx2.v2a_cross_attn"):
                    ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                    vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a

                    # Strict-Ulysses A2A pattern (video → audio). Padded Q
                    # rows (tail of audio) produce garbage that is stripped
                    # by LTXModel.forward on return, so v2a takes no
                    # key_padding_mask.
                    k_v2a, v_v2a = self.video_to_audio_attn.project_kv(vx_scaled)

                    v2a_out = (
                        self.video_to_audio_attn(
                            ax_scaled,
                            pre_projected_kv=(k_v2a, v_v2a),
                            pe=audio.cross_positional_embeddings,
                            k_pe=video.cross_positional_embeddings,
                        )
                        * gate_out_v2a
                    )
                    if has_perturbations and perturbations.any_in_batch(
                        PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx
                    ):
                        v2a_out = v2a_out * perturbations.mask_like(
                            PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, v2a_out
                        )
                    ax = ax + v2a_out

        # --- Video FFN ---
        if run_vx:
            with nvtx_range("ltx2.video_ffn"):
                vshift_mlp, vscale_mlp, vgate_mlp = self._get_ada_values(
                    self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, None)
                )
                vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
                vx = vx + self.ff(vx_scaled) * vgate_mlp

        # --- Audio FFN ---
        if run_ax:
            with nvtx_range("ltx2.audio_ffn"):
                ashift_mlp, ascale_mlp, agate_mlp = self._get_ada_values(
                    self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, None)
                )
                ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
                ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )


# ---------------------------------------------------------------------------
# LTXModelType + LTXModel (top-level)
# ---------------------------------------------------------------------------


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(nn.Module):
    """LTX-2 transformer built from TRT-LLM primitives.

    Native implementation using optimized TRT-LLM Linear, RMSNorm, MLP, and
    attention backends for all compute-heavy operations.

    The architecture-specific wiring (RoPE, AdaLN, dual-stream blocks, etc.)
    follows the Lightricks reference implementation.
    """

    def __init__(
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        model_config: Optional["DiffusionModelConfig"] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.model_type = model_type
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta

        cross_pe_max_pos = None

        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(in_channels, out_channels, caption_channels, norm_eps)

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(audio_in_channels, audio_out_channels, caption_channels, norm_eps)

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(
                self.positional_embedding_max_pos[0],
                self.audio_positional_embedding_max_pos[0],
            )
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)

        vgm = model_config.visual_gen_mapping
        primary_heads = (
            num_attention_heads if model_type.is_video_enabled() else audio_num_attention_heads
        )
        ulysses_size = vgm.ulysses_size if vgm else 1
        if ulysses_size > 1 and primary_heads % ulysses_size != 0:
            raise ValueError(
                f"num_attention_heads ({primary_heads}) must be divisible by "
                f"ulysses_size ({ulysses_size})"
            )
        self.use_ulysses = ulysses_size > 1
        self.ulysses_size = ulysses_size
        self.ulysses_pg = vgm.ulysses_group if vgm else None
        self.ulysses_rank = vgm.ulysses_rank if vgm else 0
        # Audio is sharded by Ulysses only when its sequence length is
        # divisible by ulysses_size (checked at runtime in forward).
        # Head divisibility is validated here since the attention backend
        # is created at init with sharded head counts.
        if self.use_ulysses and model_type.is_audio_enabled():
            if audio_num_attention_heads % self.ulysses_size != 0:
                raise ValueError(
                    f"audio_num_attention_heads ({audio_num_attention_heads}) "
                    f"must be divisible by ulysses_size ({self.ulysses_size})"
                )

        self._audio_is_sharded = False
        # Strict-Ulysses audio padding bookkeeping. Populated by
        # configure_audio_ulysses once the raw audio sequence length is
        # known; read by forward to shape the key-padding mask and strip
        # the padded tail on exit.
        self._audio_valid_len = 0
        self._audio_pad = 0

        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=(
                audio_attention_head_dim if model_type.is_audio_enabled() else 0
            ),
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            apply_gated_attention=apply_gated_attention,
        )

        self.__post_init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def __post_init__(self):
        """Apply quant exclusions then materialize deferred Linear weights."""
        self._apply_quant_config_exclude_modules()
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    # ==================== FP8 static checkpoint workaround ====================
    # Pre-quantized FP8 checkpoints (HuggingFace _quantization_metadata format)
    # embed layer names using the original checkpoint convention, which diverges
    # from TRT-LLM model names after QKV fusion and FF remapping.
    #
    # _remap_exclude_modules translates those names so that non-quantized layers
    # are correctly excluded from FP8 quantization.
    #
    # TODO: Remove this block once checkpoint tooling emits model-convention
    # names directly (i.e. qkv_proj, up_proj, down_proj instead of
    # to_q/to_k/to_v, ff.net.0.proj, ff.net.2).
    # ========================================================================

    @staticmethod
    def _remap_exclude_modules(exclude_modules: list[str]) -> list[str]:
        """Translate checkpoint-convention exclude names to model-convention names.

        The checkpoint uses naming conventions that differ from the TRT-LLM
        model after QKV fusion and FF remapping:
          - Self-attention QKV: ``to_q / to_k / to_v`` → fused ``qkv_proj``
          - FeedForward:        ``ff.net.0.proj / ff.net.2`` → ``ff.up_proj / ff.down_proj``

        Returns a combined list containing both original and remapped patterns
        so that ``fnmatch`` can match either convention.
        """
        remapped: set[str] = set()
        for entry in exclude_modules:
            for qkv_suffix in (".to_q", ".to_k", ".to_v"):
                if entry.endswith(qkv_suffix):
                    remapped.add(entry[: -len(qkv_suffix)] + ".qkv_proj")
            for ff_prefix in (".ff.", ".audio_ff."):
                old_up = ff_prefix + "net.0.proj"
                old_down = ff_prefix + "net.2"
                if old_up in entry:
                    remapped.add(entry.replace(old_up, ff_prefix + "up_proj"))
                elif old_down in entry:
                    remapped.add(entry.replace(old_down, ff_prefix + "down_proj"))
        return list(exclude_modules) + sorted(remapped)

    # ==================== End FP8 static checkpoint workaround ===============

    def _apply_quant_config_exclude_modules(self):
        if self.model_config is None:
            return
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return

        kv_cache_quant_algo = quant_config.kv_cache_quant_algo if quant_config else None
        no_quant_config = QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)

        needs_remap = quant_config.quant_algo in (QuantAlgo.FP8,)
        if needs_remap:
            # FP8 static checkpoint: remap exclude names (see above)
            all_patterns = self._remap_exclude_modules(quant_config.exclude_modules)
        else:
            all_patterns = list(quant_config.exclude_modules)

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                is_excluded = any(fnmatch.fnmatchcase(name, pat) for pat in all_patterns)
                if is_excluded and getattr(module, "quant_config", None) is not None:
                    module.quant_config = no_quant_config

    # -- Initialization helpers ----------------------------------------------

    def _make_linear(self, in_features: int, out_features: int, bias: bool = True) -> nn.Module:
        """Create a Linear layer using the TRT-LLM backend."""
        dtype = self.model_config.torch_dtype if self.model_config else None
        quant_config = self.model_config.quant_config if self.model_config else None
        skip_create = self.model_config.skip_create_weights_in_init if self.model_config else False
        force_dq = self.model_config.force_dynamic_quantization if self.model_config else False
        mapping = getattr(self.model_config, "mapping", None) if self.model_config else None
        return Linear(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create,
            force_dynamic_quantization=force_dq,
        )

    def _init_video(self, in_channels, out_channels, caption_channels, norm_eps):
        self.patchify_proj = self._make_linear(in_channels, self.inner_dim)
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            make_linear=self._make_linear,
        )
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.inner_dim,
            make_linear=self._make_linear,
        )
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = self._make_linear(self.inner_dim, out_channels)

    def _init_audio(self, in_channels, out_channels, caption_channels, norm_eps):
        self.audio_patchify_proj = self._make_linear(in_channels, self.audio_inner_dim)
        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            make_linear=self._make_linear,
        )
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=self.audio_inner_dim,
            make_linear=self._make_linear,
        )
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(
            self.audio_inner_dim, elementwise_affine=False, eps=norm_eps
        )
        self.audio_proj_out = self._make_linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(self, num_scale_shift_values):
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
            make_linear=self._make_linear,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
            make_linear=self._make_linear,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
            make_linear=self._make_linear,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
            make_linear=self._make_linear,
        )

    def _init_preprocessors(self, cross_pe_max_pos):
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                caption_projection=self.caption_projection,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                caption_projection=self.audio_caption_projection,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
            )

    def _init_transformer_blocks(
        self,
        num_layers,
        attention_head_dim,
        cross_attention_dim,
        audio_attention_head_dim,
        audio_cross_attention_dim,
        norm_eps,
        apply_gated_attention,
    ):
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    config=self.model_config,
                )
                for idx in range(num_layers)
            ]
        )

    # -- Ulysses sequence sharding / gathering --------------------------------

    def _shard_transformer_args(self, args: TransformerArgs) -> TransformerArgs:
        """Shard sequence-dependent fields of *args* for Ulysses."""
        seq_len = args.x.shape[1]
        chunk = seq_len // self.ulysses_size
        s = self.ulysses_rank * chunk
        e = s + chunk

        def _shard(t):
            if t is None or t.ndim < 2 or t.shape[1] != seq_len:
                return t
            return t[:, s:e]

        def _shard_pe(pe):
            if pe is None:
                return None
            cos, sin = pe
            if cos.ndim == 4 and cos.shape[2] == seq_len:
                # Split RoPE: [B, H, S, D] — sequence dim at index 2
                return (cos[:, :, s:e], sin[:, :, s:e])
            if cos.ndim == 3 and cos.shape[1] == seq_len:
                # Interleaved RoPE: [B, S, D] — sequence dim at index 1
                return (cos[:, s:e], sin[:, s:e])
            raise ValueError(
                "positional-embedding sequence axis does not match the "
                f"transformer input: seq_len={seq_len}, cos.ndim={cos.ndim}, "
                f"cos.shape={tuple(cos.shape)}, sin.shape={tuple(sin.shape)}. "
                "Expected split layout [B, H, S, D] with cos.shape[2]==seq_len "
                "or interleaved layout [B, S, D] with cos.shape[1]==seq_len. "
                "An unsharded pe would silently desynchronize RoPE across "
                "Ulysses ranks and corrupt downstream attention; the caller "
                "must rebuild the positional embedding at the new sequence "
                "layout before sharding."
            )

        return replace(
            args,
            x=args.x[:, s:e],
            timesteps=_shard(args.timesteps),
            embedded_timestep=_shard(args.embedded_timestep),
            positional_embeddings=_shard_pe(args.positional_embeddings),
            cross_positional_embeddings=_shard_pe(args.cross_positional_embeddings),
            cross_scale_shift_timestep=_shard(args.cross_scale_shift_timestep),
            cross_gate_timestep=_shard(args.cross_gate_timestep),
            # audio_padding_mask is full-seq [B, S_full] and identical across
            # ranks by construction. After the K/V a2a in
            # UlyssesCrossAttention each rank holds the full K/V seq and
            # applies the mask locally in SDPA, so no sharding is required.
            audio_padding_mask=args.audio_padding_mask,
        )

    def _gather_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather hidden states along the sequence dim."""
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(self.ulysses_size)]
        dist.all_gather(gathered, x, group=self.ulysses_pg)
        return torch.cat(gathered, dim=1)

    def configure_audio_ulysses(self, audio_seq_len: int) -> None:
        """Configure audio Ulysses sharding for the given raw audio length.

        Called once before the denoising loop when the audio token count is
        known. Under strict-Ulysses, audio is ALWAYS sharded when Ulysses is
        enabled; non-divisible raw lengths are handled by padding to the next
        multiple of ``ulysses_size`` in ``LTXModel.forward`` and masking
        padded slots in ``audio_attn1`` / ``audio_to_video_attn``. The cached
        ``_audio_pad`` is read by ``forward`` to shape the mask and strip on
        exit. ``set_ulysses_enabled(False)`` still overrides by clearing
        ``_audio_is_sharded`` so Stage 2 runs on the plain inner backends.
        """
        if not self.use_ulysses:
            self._audio_is_sharded = False
            self._audio_valid_len = audio_seq_len
            self._audio_pad = 0
            return

        U = self.ulysses_size
        self._audio_is_sharded = True
        self._audio_valid_len = audio_seq_len
        self._audio_pad = (U - audio_seq_len % U) % U
        for block in self.transformer_blocks:
            block._audio_is_sharded = True
            if hasattr(block, "audio_attn1"):
                block.audio_attn1.set_ulysses_active(True)

    def set_ulysses_enabled(self, enabled: bool) -> None:
        """Enable or disable Ulysses parallelism at runtime.

        Call with ``False`` before running the transformer on a single rank
        (e.g. Stage 2 of the two-stage pipeline where non-primary workers
        have already exited). Call with ``True`` to restore multi-rank
        operation; audio sharding will be reconfigured by the next
        :meth:`configure_audio_ulysses` call.

        Under strict-Ulysses the AV cross-attention modules
        (``audio_to_video_attn``, ``video_to_audio_attn``) are Ulysses-aware
        through ``UlyssesCrossAttention``, so this method toggles their
        dual-attn path alongside ``attn1`` and ``audio_attn1``.
        """
        if self.ulysses_size <= 1:
            return

        self.use_ulysses = enabled
        if not enabled:
            self._audio_is_sharded = False

        for block in self.transformer_blocks:
            block._use_ulysses = enabled
            if not enabled:
                block._audio_is_sharded = False
            if hasattr(block, "attn1"):
                block.attn1.set_ulysses_active(enabled)
            if hasattr(block, "audio_attn1"):
                block.audio_attn1.set_ulysses_active(enabled)
            if hasattr(block, "audio_to_video_attn"):
                block.audio_to_video_attn.set_ulysses_active(enabled)
            if hasattr(block, "video_to_audio_attn"):
                block.video_to_audio_attn.set_ulysses_active(enabled)

    # -- Output processing ---------------------------------------------------

    @staticmethod
    def _process_output(
        scale_shift_table: nn.Parameter,
        norm_out: nn.LayerNorm,
        proj_out: nn.Module,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = norm_out(x)
        x = x * (1 + scale) + shift
        return proj_out(x)

    # -- Audio padding helpers (strict-Ulysses) --------------------------------

    @staticmethod
    def _pad_modality_audio(audio: Modality, pad: int) -> Modality:
        """Pad ``audio`` on the token axis by ``pad`` slots.

        - ``latent``: zero-pad on ``dim=1`` (tokens pass through per-row norm/MLP cleanly).
        - ``positions``: repeat-last on ``dim=2`` so RoPE cos/sin for padded
          slots equal the last-valid slot's values exactly, keeping positions
          inside the model's trained range regardless of
          ``positional_embedding_max_pos``. Works for shapes
          ``(B, n_dims, T)`` and ``(B, n_dims, T, 2)``.
        - ``timesteps``: repeat-last on ``dim=1`` only when the tensor is
          per-token ``(B, T)``; scalar ``(B,)`` timesteps are not padded.
        - ``context`` / ``context_mask``: untouched (text-side, not audio-
          token-count-dependent).
        """
        if pad <= 0:
            return audio

        latent = F.pad(audio.latent, (0, 0, 0, pad))  # Pad dim=1 with zeros.

        pos = audio.positions
        last = pos[:, :, -1:, ...]
        repeat_shape = list(pos.shape)
        repeat_shape[2] = pad
        tail = last.expand(*repeat_shape).contiguous()
        positions = torch.cat([pos, tail], dim=2)

        # Per-token timesteps have shape (B, T) where T matches the original
        # positions seq length (pre-padding). Repeat-last-pad on dim=1 when
        # that signature is present; scalar (B,) timesteps pass through.
        if audio.timesteps.ndim >= 2 and audio.timesteps.shape[1] == pos.shape[2]:
            ts = audio.timesteps
            last_ts = ts[:, -1:, ...].expand(
                ts.shape[0], pad, *ts.shape[2:]
            ).contiguous()
            timesteps = torch.cat([ts, last_ts], dim=1)
        else:
            timesteps = audio.timesteps

        # Modality is frozen; construct a new instance rather than replacing.
        from dataclasses import replace as _dc_replace  # local to avoid shadow.
        return _dc_replace(
            audio, latent=latent, positions=positions, timesteps=timesteps
        )

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass through the LTX-2 transformer.

        Args:
            video: Video modality input (or None).
            audio: Audio modality input (or None).
            perturbations: Optional ``BatchedPerturbationConfig`` for STG.

        Returns:
            Tuple of (video_output, audio_output) velocity predictions.
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        # Strict-Ulysses audio padding: pad on entry so S_full % U == 0,
        # build a [B, S_full] validity mask, and strip the padded tail on
        # return. Disabled when Ulysses is off (padding owned by
        # configure_audio_ulysses; _audio_pad == 0 in that case).
        audio_pad = getattr(self, "_audio_pad", 0) if self.use_ulysses else 0
        audio_padding_mask = None
        if audio is not None and audio_pad > 0:
            s_real = audio.latent.shape[1]
            audio = self._pad_modality_audio(audio, audio_pad)
            s_full = audio.latent.shape[1]
            assert s_full == s_real + audio_pad
            if self.ulysses_size > 1:
                assert s_full % self.ulysses_size == 0, (
                    f"Padded audio length {s_full} must be divisible by "
                    f"ulysses_size {self.ulysses_size}"
                )
            audio_padding_mask = torch.ones(
                audio.latent.shape[0],
                s_full,
                dtype=torch.bool,
                device=audio.latent.device,
            )
            audio_padding_mask[:, s_real:] = False
        audio_s_real = audio.latent.shape[1] - audio_pad if audio is not None else 0

        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio) if audio is not None else None

        if audio_args is not None and audio_padding_mask is not None:
            audio_args = replace(audio_args, audio_padding_mask=audio_padding_mask)

        # Shard sequences for Ulysses parallelism.
        # Video is always sharded. Audio is sharded whenever Ulysses is on
        # (padded up to a multiple of U in the block above); Stage 2
        # overrides this by flipping _audio_is_sharded via
        # set_ulysses_enabled(False).
        if self.use_ulysses:
            if video_args is not None:
                video_args = self._shard_transformer_args(video_args)
            if self._audio_is_sharded and audio_args is not None:
                audio_args = self._shard_transformer_args(audio_args)

        for block in self.transformer_blocks:
            video_args, audio_args = block(
                video=video_args,
                audio=audio_args,
                perturbations=perturbations,
            )

        # Gather sequences back to full length for output processing.
        # Only gather embedded_timestep if it was actually sharded (dim-1
        # matches x); scalar timestep embeddings [B, 1, D] are
        # broadcast-compatible and must not be gathered.
        if self.use_ulysses:
            if video_args is not None:
                gathered_vx = self._gather_sequence(video_args.x)
                v_et = video_args.embedded_timestep
                if v_et.shape[1] == video_args.x.shape[1]:
                    v_et = self._gather_sequence(v_et)
                video_args = replace(
                    video_args,
                    x=gathered_vx,
                    embedded_timestep=v_et,
                )
            if self._audio_is_sharded and audio_args is not None:
                gathered_ax = self._gather_sequence(audio_args.x)
                a_et = audio_args.embedded_timestep
                if a_et.shape[1] == audio_args.x.shape[1]:
                    a_et = self._gather_sequence(a_et)
                audio_args = replace(
                    audio_args,
                    x=gathered_ax,
                    embedded_timestep=a_et,
                )

        vx = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_args.x,
                video_args.embedded_timestep,
            )
            if video_args is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_args.x,
                audio_args.embedded_timestep,
            )
            if audio_args is not None
            else None
        )
        # Strip the padded tail from the audio output so the caller sees
        # the original S_real it passed in. Must happen AFTER _process_output
        # so that embedded_timestep (which may be scalar [B, 1, D]) is not
        # accidentally cropped.
        if ax is not None and audio_pad > 0:
            ax = ax[:, :audio_s_real, :]
        return vx, ax

    # -- Weight loading (from a single LTX-2 .safetensors checkpoint) -------------------------

    def load_weights(self, weights: dict) -> None:
        """Load checkpoint weights with key remapping.

        Handles naming differences between checkpoint and model:
          FFN:    ``ff.net.0.proj.*`` / ``ff.net.2.*`` → ``ff.up_proj.*`` / ``ff.down_proj.*``
          QKNorm: ``*.q_norm.*`` / ``*.k_norm.*``      → ``*.norm_q.*``   / ``*.norm_k.*``
        """
        remapped = {}
        for key, value in weights.items():
            new_key = key
            for ff_prefix in (".ff.", ".audio_ff."):
                if ff_prefix + "net.0.proj." in new_key:
                    new_key = new_key.replace(ff_prefix + "net.0.proj.", ff_prefix + "up_proj.")
                elif ff_prefix + "net.2." in new_key:
                    new_key = new_key.replace(ff_prefix + "net.2.", ff_prefix + "down_proj.")
            new_key = new_key.replace(".q_norm.", ".norm_q.")
            new_key = new_key.replace(".k_norm.", ".norm_k.")
            remapped[new_key] = value
        weights = remapped

        target_dtype = self.model_config.torch_dtype if self.model_config else torch.bfloat16

        model_keys = {
            (name + "." + pname) if name else pname
            for name, mod in self.named_modules()
            for pname, p in mod._parameters.items()
            if p is not None
        }
        checkpoint_keys = set(weights.keys())

        # FUSE_QKV self-attention: model has qkv_proj, checkpoint has
        # to_q/to_k/to_v.  The weight loader fuses them via params_map.
        # Exclude these from mismatch warnings.
        fused_model_params = set()
        fused_ckpt_params = set()
        for name, mod in self.named_modules():
            if isinstance(mod, Linear):
                wlc = getattr(mod, "weights_loading_config", None)
                if wlc and getattr(wlc, "weight_mode", None) == WeightMode.FUSED_QKV_LINEAR:
                    parent = ".".join(name.split(".")[:-1])
                    for pname, p in mod._parameters.items():
                        if p is not None:
                            fused_model_params.add(f"{name}.{pname}")
                            for src in ("to_q", "to_k", "to_v"):
                                fused_ckpt_params.add(f"{parent}.{src}.{pname}")

        missing = (model_keys - checkpoint_keys) - fused_model_params
        unexpected = (checkpoint_keys - model_keys) - fused_ckpt_params
        quantized = (
            self.model_config is not None and self.model_config.quant_config.quant_algo is not None
        )
        dynamic_weight_quant = (
            self.model_config is not None and self.model_config.dynamic_weight_quant
        )
        if missing:
            logger.warning(
                f"LTXModel: {len(missing)} model params NOT in checkpoint: "
                f"{sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}"
            )
        if unexpected:
            logger.warning(
                f"LTXModel: {len(unexpected)} checkpoint keys NOT in model: "
                f"{sorted(unexpected)[:20]}{'...' if len(unexpected) > 20 else ''}"
            )
        loaded = model_keys & checkpoint_keys
        logger.info(
            f"LTXModel weight check: {len(loaded)} matched, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
        if quantized and missing:
            if dynamic_weight_quant:
                logger.info(
                    "Dynamic quantization is enabled -- missing scale parameters "
                    "(e.g. weight_scale, input_scale) are expected and will be "
                    "computed by DynamicLinearWeightLoader during weight loading."
                )
            else:
                logger.info(
                    "Pre-quantized checkpoint -- missing parameters "
                    "(e.g. alpha, inv_input_scale, kv_scales) are derived from "
                    "checkpoint scales during Linear.load_weights()."
                )

        for param_name, param in self._parameters.items():
            if param is not None and param_name in weights:
                param.data.copy_(weights[param_name].to(target_dtype))

        self._load_weights_trtllm(weights, target_dtype)

    def _load_weights_trtllm(self, weights: dict, target_dtype: torch.dtype) -> None:
        """TRT-LLM weight loading with dynamic quantization support."""
        params_map = {
            "qkv_proj": ["to_q", "to_k", "to_v"],
        }
        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)

        for name, module in tqdm(self.named_modules(), desc="Loading LTXModel weights"):
            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                if weight_dicts:
                    loader.load_linear_weights(module, name, weight_dicts)
            else:
                module_weights = loader.filter_weights(name, weights)
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(module_weights[param_name].to(target_dtype))

    def post_load_weights(self) -> None:
        """Post-load hooks: finalize quantized Linear layers."""
        for _, module in self.named_modules():
            if isinstance(module, Linear) and hasattr(module, "post_load_weights"):
                module.post_load_weights()
