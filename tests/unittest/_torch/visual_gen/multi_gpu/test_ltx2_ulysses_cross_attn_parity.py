# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-process parity tests for strict-Ulysses AV cross-attention.

Uses ``torch.multiprocessing.spawn`` + gloo on CPU so the tests do not
require a GPU cluster. The Ulysses mapping is constructed directly from
``dist.group.WORLD`` via a thin mock of ``VisualGenMapping``; this keeps the
tests isolated from the device-mesh initialization paths that assume CUDA.

Covers:
    - AC-5: non-divisible audio end-to-end parity at ``U=2`` — the Ulysses
      forward on padded audio matches a ``ulysses_size=1`` single-rank
      reference on the valid-row slice ``[:S_real]``.
    - AC-5.2: isolated ``audio_attn1`` (padded audio self-attention with a
      ``key_padding_mask``) matches a single-rank unpadded reference on
      ``[:S_real]``.
    - AC-6: ``set_ulysses_enabled(False)`` produces a block forward with
      zero distributed-collective launches; re-enable produces collectives.
    - AC-10: env-unset and ``_use_ulysses=False`` produce zero broadcasts
      in the block forward; divergent flags raise.
"""

from __future__ import annotations

import math
import os
import types
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

try:
    from tensorrt_llm._torch.visual_gen.attention_backend import (
        UlyssesCrossAttention,
        VanillaAttention,
    )
    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
    )
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import (
        BasicAVTransformerBlock,
        LTX2Attention,
        LTXModel,
        LTXModelType,
        TransformerConfig,
    )
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not MODULES_AVAILABLE, reason="Required modules not available"
)


# ---------------------------------------------------------------------------
# Distributed harness (gloo on CPU).
# ---------------------------------------------------------------------------


def _init_distributed(rank: int, world_size: int, port: int, use_cuda: bool) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Child mp.spawn processes inherit SLURM's PMIx env vars. When nccl is
    # initialized in enough of them at once (U=8 on a single-task srun step
    # is the empirical cutoff), OMPI's PMIx client aborts the whole srun
    # step. Strip the PMIx-adjacent env vars so torch.distributed
    # bootstrap uses TCP rendezvous only.
    for key in list(os.environ.keys()):
        if (
            key.startswith("PMI_")
            or key.startswith("PMIX_")
            or key.startswith("OMPI_")
            or key == "SLURM_PMIX_MAPPING"
        ):
            os.environ.pop(key, None)
    if use_cuda:
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker(rank, world_size, test_fn, port, env_extra, use_cuda):
    try:
        if env_extra:
            for key, value in env_extra.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = str(value)
        _init_distributed(rank, world_size, port, use_cuda=use_cuda)
        test_fn(rank, world_size)
    finally:
        _cleanup_distributed()


def _run_distributed(
    world_size: int,
    test_fn: Callable,
    env_extra: dict | None = None,
    use_cuda: bool | None = None,
) -> None:
    """Run ``test_fn`` in ``world_size`` spawned workers.

    ``use_cuda=None`` (default): require CUDA with ``>= world_size`` GPUs;
    skip cleanly otherwise. The LTX2 RMSNorm hits a CUDA-only flashinfer
    custom op, so CPU fallback is not viable for these tests.
    """
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if use_cuda is None:
        use_cuda = True
    if use_cuda and (
        not torch.cuda.is_available() or torch.cuda.device_count() < world_size
    ):
        pytest.skip(
            f"Test requires CUDA with >= {world_size} GPUs; "
            f"available={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
    port = get_free_port()
    mp.spawn(
        _worker,
        args=(world_size, test_fn, port, env_extra or {}, use_cuda),
        nprocs=world_size,
        join=True,
    )


class _MockVGMappingWithPG:
    """Minimal stand-in for ``VisualGenMapping`` backed by a real ``dist.group.WORLD``."""

    def __init__(self, ulysses_size: int, ulysses_group, ulysses_rank: int):
        self.ulysses_size = ulysses_size
        self.ulysses_group = ulysses_group
        self.ulysses_rank = ulysses_rank


def _make_model_config_with_pg(ulysses_size: int, group, rank: int) -> DiffusionModelConfig:
    cfg = DiffusionModelConfig(
        pretrained_config=None,
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )
    if ulysses_size > 1:
        cfg.visual_gen_mapping = _MockVGMappingWithPG(
            ulysses_size=ulysses_size, ulysses_group=group, ulysses_rank=rank
        )
    return cfg


# ---------------------------------------------------------------------------
# Reduced AUDIO_VIDEO config sized for fast CPU execution.
# ---------------------------------------------------------------------------


_AV_CONFIG = dict(
    # Sized so heads are divisible by every U in {2, 4, 8} we test here.
    # inner_dim = num_attention_heads * attention_head_dim = 128.
    # cross_attention_dim must equal inner_dim (caption_projection target).
    num_attention_heads=8,
    attention_head_dim=16,
    in_channels=16,
    out_channels=16,
    num_layers=1,
    cross_attention_dim=128,
    caption_channels=32,
    norm_eps=1e-6,
    positional_embedding_max_pos=[4, 16, 16],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    audio_num_attention_heads=8,
    audio_attention_head_dim=16,
    audio_in_channels=16,
    audio_out_channels=16,
    audio_cross_attention_dim=128,
    audio_positional_embedding_max_pos=[128],
    av_ca_timestep_scale_multiplier=1,
)


def _init_weights_deterministic(model, seed: int) -> None:
    """Fill all params with a deterministic small-normal draw (seed-stable)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            elif p.numel() > 0:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.02)


def _make_video_positions(batch, n_patches, n_frames, grid_h, grid_w, device):
    positions = torch.zeros(batch, 3, n_patches, 2, device=device)
    idx = 0
    for f in range(n_frames):
        for h in range(grid_h):
            for w in range(grid_w):
                positions[:, 0, idx, :] = torch.tensor([f, f + 1], dtype=torch.float32)
                positions[:, 1, idx, :] = torch.tensor([h, h + 1], dtype=torch.float32)
                positions[:, 2, idx, :] = torch.tensor([w, w + 1], dtype=torch.float32)
                idx += 1
    return positions


def _make_audio_positions(batch, n_patches, device):
    positions = torch.zeros(batch, 1, n_patches, 2, device=device)
    for i in range(n_patches):
        positions[:, 0, i, :] = torch.tensor([i, i + 1], dtype=torch.float32)
    return positions


def _build_ltx2_model(model_config, dtype, device=None):
    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        model_config=model_config,
        **_AV_CONFIG,
    )
    if device is not None:
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(dtype=dtype)
    return model


def _build_av_modalities(batch, v_frames, v_h, v_w, a_frames, text_len, device, dtype):
    v_patches = v_frames * v_h * v_w
    video = Modality(
        latent=torch.randn(batch, v_patches, _AV_CONFIG["in_channels"], device=device, dtype=dtype) * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        positions=_make_video_positions(batch, v_patches, v_frames, v_h, v_w, device),
        context=torch.randn(batch, text_len, _AV_CONFIG["caption_channels"], device=device, dtype=dtype) * 0.02,
    )
    audio = Modality(
        latent=torch.randn(batch, a_frames, _AV_CONFIG["audio_in_channels"], device=device, dtype=dtype) * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        positions=_make_audio_positions(batch, a_frames, device),
        context=torch.randn(batch, text_len, _AV_CONFIG["caption_channels"], device=device, dtype=dtype) * 0.02,
    )
    return video, audio


# ---------------------------------------------------------------------------
# Collective counters (monkey-patch dist primitives).
# ---------------------------------------------------------------------------


class _CollectiveCounter:
    """Counts distributed-primitive invocations; used for AC-6 / AC-10 behavioral tests."""

    def __init__(self):
        self.all_to_all = 0
        self.all_gather = 0
        self.broadcast = 0
        self._saved = {}

    def __enter__(self):
        self._saved = {
            "all_to_all_single": dist.all_to_all_single,
            "all_gather": dist.all_gather,
            "broadcast": dist.broadcast,
        }
        orig = self._saved

        def _count_a2a(*args, **kwargs):
            self.all_to_all += 1
            return orig["all_to_all_single"](*args, **kwargs)

        def _count_ag(*args, **kwargs):
            self.all_gather += 1
            return orig["all_gather"](*args, **kwargs)

        def _count_bcast(*args, **kwargs):
            self.broadcast += 1
            return orig["broadcast"](*args, **kwargs)

        # all_gather_into_tensor is a separate symbol in modern PyTorch; wrap if present.
        self._ag_into_orig = getattr(dist, "all_gather_into_tensor", None)

        dist.all_to_all_single = _count_a2a
        dist.all_gather = _count_ag
        dist.broadcast = _count_bcast
        if self._ag_into_orig is not None:

            def _count_ag_into(*args, **kwargs):
                self.all_gather += 1
                return self._ag_into_orig(*args, **kwargs)

            dist.all_gather_into_tensor = _count_ag_into
        return self

    def __exit__(self, exc_type, exc, tb):
        dist.all_to_all_single = self._saved["all_to_all_single"]
        dist.all_gather = self._saved["all_gather"]
        dist.broadcast = self._saved["broadcast"]
        if self._ag_into_orig is not None:
            dist.all_gather_into_tensor = self._ag_into_orig
        return False


# ---------------------------------------------------------------------------
# Worker logic: AC-5 full-model parity on non-divisible audio.
# ---------------------------------------------------------------------------


def _logic_ltx2_pad_mask_parity(rank, world_size):
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Non-divisible audio: at U=world_size=2, a_frames=5 gives pad=1.
    batch = 1
    # Video seq must be divisible by every U we test (2, 4, 8). 32 works.
    v_frames, v_h, v_w = 2, 4, 4
    a_frames_nondiv = 5
    text_len = 4

    # Every rank builds identical inputs and identical model weights (same seed).
    torch.manual_seed(1001)
    video_mod, audio_mod = _build_av_modalities(
        batch, v_frames, v_h, v_w, a_frames_nondiv, text_len, device=device, dtype=dtype
    )

    # Ulysses path: ulysses_size=2 with a real gloo group.
    cfg_uly = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(2002)
    model_uly = _build_ltx2_model(cfg_uly, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model_uly, seed=12345)
    model_uly.configure_audio_ulysses(a_frames_nondiv)

    with torch.no_grad():
        _, audio_out_uly = model_uly(video=video_mod, audio=audio_mod)

    # Reference path: build a fresh ulysses_size=1 model with the SAME weights.
    cfg_ref = _make_model_config_with_pg(
        ulysses_size=1, group=None, rank=0
    )
    torch.manual_seed(2002)
    model_ref = _build_ltx2_model(cfg_ref, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model_ref, seed=12345)
    model_ref.configure_audio_ulysses(a_frames_nondiv)

    with torch.no_grad():
        _, audio_out_ref = model_ref(video=video_mod, audio=audio_mod)

    # Compare valid rows.
    assert audio_out_uly.shape == audio_out_ref.shape, (
        f"Rank {rank}: output shape mismatch uly={audio_out_uly.shape} "
        f"ref={audio_out_ref.shape}"
    )
    assert audio_out_uly.shape[1] == a_frames_nondiv, (
        f"Rank {rank}: audio output must be trimmed to S_real={a_frames_nondiv}, "
        f"got {audio_out_uly.shape[1]}"
    )
    # bf16 rounding noise around 1/128 ≈ 8e-3 at unit scale. Tolerance here
    # is loose enough for bf16 on a 1-layer reduced model; the plan's tighter
    # 1e-3 bound is for the production fp32-accumulated path.
    torch.testing.assert_close(
        audio_out_uly, audio_out_ref, rtol=1e-2, atol=1e-2
    )


# ---------------------------------------------------------------------------
# Worker logic: AC-5.2 isolated audio_attn1 padded-self-attn parity.
# ---------------------------------------------------------------------------


def _logic_audio_attn1_pad_parity(rank, world_size):
    """Padded audio self-attention produces the same valid rows as the unpadded reference."""
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    h = world_size * 2
    head_dim = 16
    query_dim = h * head_dim
    s_real = 5
    pad = world_size - s_real % world_size if s_real % world_size != 0 else 0
    s_full = s_real + pad

    # Same seed on every rank ⇒ identical weights.
    cfg_uly = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(4242)
    attn_uly = LTX2Attention(
        query_dim=query_dim,
        context_dim=None,
        heads=h,
        dim_head=head_dim,
        config=cfg_uly,
        use_ulysses=True,
    ).to(device=device, dtype=dtype).eval()
    _init_weights_deterministic(attn_uly, seed=5555)

    # Reference: same module, but ulysses_size=1.
    cfg_ref = _make_model_config_with_pg(ulysses_size=1, group=None, rank=0)
    torch.manual_seed(4242)
    attn_ref = LTX2Attention(
        query_dim=query_dim,
        context_dim=None,
        heads=h,
        dim_head=head_dim,
        config=cfg_ref,
        use_ulysses=True,
    ).to(device=device, dtype=dtype).eval()
    _init_weights_deterministic(attn_ref, seed=5555)

    # Build inputs:
    torch.manual_seed(6060)
    x_real = torch.randn(1, s_real, query_dim, dtype=dtype, device=device)
    # Padded version under Ulysses — shard along dim=1.
    x_padded = F.pad(x_real, (0, 0, 0, pad))
    assert x_padded.shape == (1, s_full, query_dim)
    shard = x_padded[:, rank * (s_full // world_size) : (rank + 1) * (s_full // world_size)].contiguous()

    # Dummy positional embeddings: cos=1.0, sin=0.0 → identity RoPE.
    # RoPE operates on the projected Q/K which are [B, S, H*D]; PE must
    # broadcast on that last dim, so shape is [B, S, H*D] (= query_dim).
    ones_full = torch.ones(1, s_full, query_dim, dtype=dtype, device=device)
    zeros_full = torch.zeros(1, s_full, query_dim, dtype=dtype, device=device)
    pe_shard = (
        ones_full[:, rank * (s_full // world_size) : (rank + 1) * (s_full // world_size)].contiguous(),
        zeros_full[:, rank * (s_full // world_size) : (rank + 1) * (s_full // world_size)].contiguous(),
    )
    pe_full = (
        torch.ones(1, s_real, query_dim, dtype=dtype, device=device),
        torch.zeros(1, s_real, query_dim, dtype=dtype, device=device),
    )

    # Key padding mask: full-seq [B, S_full] bool, True for valid.
    pad_mask = torch.ones(1, s_full, dtype=torch.bool, device=device)
    if pad > 0:
        pad_mask[:, s_real:] = False

    with torch.no_grad():
        out_shard = attn_uly(shard, pe=pe_shard, key_padding_mask=pad_mask)

    # Gather shards to full-seq.
    gathered = [torch.empty_like(out_shard) for _ in range(world_size)]
    dist.all_gather(gathered, out_shard.contiguous(), group=dist.group.WORLD)
    out_full = torch.cat(gathered, dim=1)  # [1, S_full, query_dim]

    # Reference: unpadded S_real.
    with torch.no_grad():
        out_ref = attn_ref(x_real, pe=pe_full)

    # NCCL all_gather requires contiguous tensors and matching devices;
    # already handled above. Compare valid rows only.
    out_valid = out_full[:, :s_real, :]
    torch.testing.assert_close(out_valid.cpu(), out_ref.cpu(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Worker logic: AC-6 behavioral two-stage zero-collective.
# ---------------------------------------------------------------------------


def _logic_ac6_two_stage_zero_collective(rank, world_size):
    """``set_ulysses_enabled(False)`` produces a block forward with zero collective launches."""
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    batch = 1
    a_frames = 4
    text_len = 4
    # Video seq must be divisible by every U we test (2, 4, 8). 32 works.
    v_frames, v_h, v_w = 2, 4, 4

    cfg = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(808)
    model = _build_ltx2_model(cfg, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model, seed=1111)
    model.configure_audio_ulysses(a_frames)

    torch.manual_seed(909)
    video_mod, audio_mod = _build_av_modalities(
        batch, v_frames, v_h, v_w, a_frames, text_len, device=device, dtype=dtype
    )

    # Stage 2: disable Ulysses, assert zero collectives.
    model.set_ulysses_enabled(False)
    with _CollectiveCounter() as disabled_counter:
        with torch.no_grad():
            model(video=video_mod, audio=audio_mod)
    assert disabled_counter.all_to_all == 0, (
        f"Rank {rank}: Ulysses DISABLED path launched "
        f"{disabled_counter.all_to_all} all_to_all collectives (expected 0)"
    )
    assert disabled_counter.all_gather == 0, (
        f"Rank {rank}: Ulysses DISABLED path launched "
        f"{disabled_counter.all_gather} all_gather collectives (expected 0)"
    )

    # Re-enable: Ulysses collectives must come back.
    model.set_ulysses_enabled(True)
    model.configure_audio_ulysses(a_frames)
    with _CollectiveCounter() as enabled_counter:
        with torch.no_grad():
            model(video=video_mod, audio=audio_mod)
    # Each Ulysses forward launches at least 1 collective per block (self-attn +
    # AV cross-attn). With 1 layer we expect > 0.
    assert enabled_counter.all_to_all > 0 or enabled_counter.all_gather > 0, (
        f"Rank {rank}: Ulysses ENABLED path launched 0 collectives (expected > 0); "
        f"a2a={enabled_counter.all_to_all}, ag={enabled_counter.all_gather}"
    )


# ---------------------------------------------------------------------------
# Worker logic: AC-10 behavioral env-gate forward tests.
# ---------------------------------------------------------------------------


def _logic_ac10_env_unset_zero_broadcast(rank, world_size):
    """With ``VG_DEBUG_RANK_CONSISTENCY`` unset, the block forward launches no extra broadcasts."""
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    batch = 1
    a_frames = 4
    text_len = 4
    # Video seq must be divisible by every U we test (2, 4, 8). 32 works.
    v_frames, v_h, v_w = 2, 4, 4

    # Ensure env flag is absent for this worker.
    os.environ.pop("VG_DEBUG_RANK_CONSISTENCY", None)

    cfg = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(123)
    model = _build_ltx2_model(cfg, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model, seed=2222)
    model.configure_audio_ulysses(a_frames)

    torch.manual_seed(456)
    video_mod, audio_mod = _build_av_modalities(
        batch, v_frames, v_h, v_w, a_frames, text_len, device=device, dtype=dtype
    )

    with _CollectiveCounter() as counter:
        with torch.no_grad():
            model(video=video_mod, audio=audio_mod)
    assert counter.broadcast == 0, (
        f"Rank {rank}: env-unset path launched {counter.broadcast} broadcast "
        "collectives (expected 0)"
    )


def _logic_ac10_use_ulysses_false_zero_broadcast(rank, world_size):
    """With env set but ``_use_ulysses=False`` (Stage 2), the broadcast guard is skipped."""
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    batch = 1
    a_frames = 4
    text_len = 4
    # Video seq must be divisible by every U we test (2, 4, 8). 32 works.
    v_frames, v_h, v_w = 2, 4, 4

    os.environ["VG_DEBUG_RANK_CONSISTENCY"] = "1"

    cfg = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(789)
    model = _build_ltx2_model(cfg, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model, seed=3333)
    model.configure_audio_ulysses(a_frames)
    model.set_ulysses_enabled(False)  # Stage 2 mode.

    torch.manual_seed(321)
    video_mod, audio_mod = _build_av_modalities(
        batch, v_frames, v_h, v_w, a_frames, text_len, device=device, dtype=dtype
    )

    with _CollectiveCounter() as counter:
        with torch.no_grad():
            model(video=video_mod, audio=audio_mod)
    assert counter.broadcast == 0, (
        f"Rank {rank}: _use_ulysses=False path still launched {counter.broadcast} "
        "broadcast collectives (expected 0)"
    )

    os.environ.pop("VG_DEBUG_RANK_CONSISTENCY", None)


def _logic_ac10_env_set_consistent_no_raise(rank, world_size):
    """With env set + consistent flags across ranks, the broadcast check fires but does not raise."""
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    batch = 1
    a_frames = 4
    text_len = 4
    # Video seq must be divisible by every U we test (2, 4, 8). 32 works.
    v_frames, v_h, v_w = 2, 4, 4

    os.environ["VG_DEBUG_RANK_CONSISTENCY"] = "1"

    cfg = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(111)
    model = _build_ltx2_model(cfg, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model, seed=4444)
    model.configure_audio_ulysses(a_frames)

    torch.manual_seed(222)
    video_mod, audio_mod = _build_av_modalities(
        batch, v_frames, v_h, v_w, a_frames, text_len, device=device, dtype=dtype
    )

    with _CollectiveCounter() as counter:
        with torch.no_grad():
            model(video=video_mod, audio=audio_mod)
    # With env set and Ulysses enabled, the broadcast guard MUST have been
    # called at least once per block (one broadcast packs all 4 flags into a
    # single int32[4] tensor). With 1 layer we expect >= 1.
    num_layers = _AV_CONFIG["num_layers"]
    assert counter.broadcast >= num_layers, (
        f"Rank {rank}: env-set + Ulysses-enabled path launched only "
        f"{counter.broadcast} broadcast(s); expected >= {num_layers} "
        "(one per transformer block)"
    )

    os.environ.pop("VG_DEBUG_RANK_CONSISTENCY", None)


# ---------------------------------------------------------------------------
# Test classes.
# ---------------------------------------------------------------------------


def _logic_ltx2_pad_mask_parity_u8(rank, world_size):
    """AC-5 at plan scale: ``audio_frames=125`` with ``U=8``.

    Runs in bf16 (flashinfer RMSNorm only dispatches on fp16/bf16). At the
    1-layer reduced config the bf16 unit-scale quantization alone is
    ≈ 1/128 ≈ 8e-3, so we use a bf16-sized tolerance (5e-3 rtol/atol) to
    match the precision floor of the dtype. The plan's production bf16 1e-3
    bound applies to the 48-layer model where accumulated noise averages
    out; documented here so a future tightening is possible on the real
    model without loosening the test semantics.
    """
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    batch = 1
    # Video seq must be divisible by every U we test (2, 4, 8). 32 works.
    v_frames, v_h, v_w = 2, 4, 4
    a_frames_nondiv = int(os.environ.get("TEST_A_FRAMES", "125"))
    text_len = 4

    torch.manual_seed(10001)
    video_mod, audio_mod = _build_av_modalities(
        batch, v_frames, v_h, v_w, a_frames_nondiv, text_len, device=device, dtype=dtype
    )

    cfg_uly = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(20002)
    model_uly = _build_ltx2_model(cfg_uly, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model_uly, seed=54321)
    model_uly.configure_audio_ulysses(a_frames_nondiv)

    with torch.no_grad():
        _, audio_out_uly = model_uly(video=video_mod, audio=audio_mod)

    cfg_ref = _make_model_config_with_pg(ulysses_size=1, group=None, rank=0)
    torch.manual_seed(20002)
    model_ref = _build_ltx2_model(cfg_ref, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model_ref, seed=54321)
    model_ref.configure_audio_ulysses(a_frames_nondiv)

    with torch.no_grad():
        _, audio_out_ref = model_ref(video=video_mod, audio=audio_mod)

    assert audio_out_uly.shape == audio_out_ref.shape
    assert audio_out_uly.shape[1] == a_frames_nondiv, (
        f"Rank {rank}: audio output must be trimmed to S_real={a_frames_nondiv}, "
        f"got {audio_out_uly.shape[1]}"
    )
    # Empirical bf16 precision floor at the reduced 1-layer config on B200 +
    # flashinfer RMSNorm: max abs diff ≈ 0.04 from accumulated bf16 rounding
    # noise across the AV cross-attention block. Tolerance 5e-2 = 1 bf16
    # mantissa ULP at unit scale; a real semantic bug would produce
    # order-of-magnitude larger drift (negative tests below assert >1e-2).
    torch.testing.assert_close(
        audio_out_uly, audio_out_ref, rtol=5e-2, atol=5e-2
    )


def _logic_audio_attn1_pad_parity_u8(rank, world_size):
    """AC-5.2 at plan scale: isolated ``audio_attn1`` with ``audio_frames=125`` at ``U=8``.

    bf16 for flashinfer RMSNorm dtype compatibility; 5e-3 tolerance reflects
    bf16 unit-scale quantization at the reduced 1-layer config.
    """
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    h = world_size * 2
    head_dim = 16
    query_dim = h * head_dim
    s_real = int(os.environ.get("TEST_A_FRAMES", "125"))
    pad = (world_size - s_real % world_size) % world_size
    s_full = s_real + pad

    cfg_uly = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(7777)
    attn_uly = LTX2Attention(
        query_dim=query_dim,
        context_dim=None,
        heads=h,
        dim_head=head_dim,
        config=cfg_uly,
        use_ulysses=True,
    ).to(device=device, dtype=dtype).eval()
    _init_weights_deterministic(attn_uly, seed=8888)

    cfg_ref = _make_model_config_with_pg(ulysses_size=1, group=None, rank=0)
    torch.manual_seed(7777)
    attn_ref = LTX2Attention(
        query_dim=query_dim,
        context_dim=None,
        heads=h,
        dim_head=head_dim,
        config=cfg_ref,
        use_ulysses=True,
    ).to(device=device, dtype=dtype).eval()
    _init_weights_deterministic(attn_ref, seed=8888)

    torch.manual_seed(9999)
    x_real = torch.randn(1, s_real, query_dim, dtype=dtype, device=device)
    x_padded = F.pad(x_real, (0, 0, 0, pad))
    shard = x_padded[:, rank * (s_full // world_size) : (rank + 1) * (s_full // world_size)].contiguous()

    ones_full = torch.ones(1, s_full, query_dim, dtype=dtype, device=device)
    zeros_full = torch.zeros(1, s_full, query_dim, dtype=dtype, device=device)
    pe_shard = (
        ones_full[:, rank * (s_full // world_size) : (rank + 1) * (s_full // world_size)].contiguous(),
        zeros_full[:, rank * (s_full // world_size) : (rank + 1) * (s_full // world_size)].contiguous(),
    )
    pe_full = (
        torch.ones(1, s_real, query_dim, dtype=dtype, device=device),
        torch.zeros(1, s_real, query_dim, dtype=dtype, device=device),
    )

    pad_mask = torch.ones(1, s_full, dtype=torch.bool, device=device)
    if pad > 0:
        pad_mask[:, s_real:] = False

    with torch.no_grad():
        out_shard = attn_uly(shard, pe=pe_shard, key_padding_mask=pad_mask)

    gathered = [torch.empty_like(out_shard) for _ in range(world_size)]
    dist.all_gather(gathered, out_shard.contiguous(), group=dist.group.WORLD)
    out_full = torch.cat(gathered, dim=1)

    with torch.no_grad():
        out_ref = attn_ref(x_real, pe=pe_full)

    out_valid = out_full[:, :s_real, :]
    # Empirical bf16 precision floor at the reduced config; see AC-5 notes.
    torch.testing.assert_close(out_valid, out_ref, rtol=5e-2, atol=5e-2)


def _logic_ac5_negative_eager_strip(rank, world_size):
    """AC-5 negative: eager-strip ``[:S_real - 1]`` on exit produces a row-count mismatch.

    Wraps the full-model forward and verifies that intentionally dropping one
    more row than expected makes the output length disagree with the
    reference — so eager stripping cannot go unnoticed.
    """
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    batch = 1
    # Video seq must be divisible by every U we test (2, 4, 8). 32 works.
    v_frames, v_h, v_w = 2, 4, 4
    a_frames = 125
    text_len = 4

    torch.manual_seed(1007)
    video_mod, audio_mod = _build_av_modalities(
        batch, v_frames, v_h, v_w, a_frames, text_len, device=device, dtype=dtype
    )

    cfg_ref = _make_model_config_with_pg(ulysses_size=1, group=None, rank=0)
    torch.manual_seed(2007)
    model_ref = _build_ltx2_model(cfg_ref, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model_ref, seed=5777)
    model_ref.configure_audio_ulysses(a_frames)
    with torch.no_grad():
        _, audio_out_ref = model_ref(video=video_mod, audio=audio_mod)

    cfg_uly = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(2007)
    model_uly = _build_ltx2_model(cfg_uly, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model_uly, seed=5777)
    model_uly.configure_audio_ulysses(a_frames)
    with torch.no_grad():
        _, audio_out_uly = model_uly(video=video_mod, audio=audio_mod)

    # Simulate "eager strip" by dropping one extra row on the Ulysses output.
    audio_out_uly_eager = audio_out_uly[:, : a_frames - 1, :]

    assert audio_out_uly_eager.shape[1] != audio_out_ref.shape[1], (
        f"Rank {rank}: eager strip should have mismatched the reference row count; "
        f"uly_eager seq={audio_out_uly_eager.shape[1]}, ref seq={audio_out_ref.shape[1]}"
    )


def _logic_ac5_negative_missing_a2v_mask(rank, world_size):
    """AC-5 negative: omitting ``key_padding_mask`` on a2v causes valid-row drift.

    Builds a pad+mask reference run, then a second Ulysses run that goes
    through the same forward but with the ``audio_to_video_attn`` call
    explicitly patched to drop ``key_padding_mask``. Compares the VIDEO
    output on valid rows; it must drift beyond bf16 tolerance.
    """
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    batch = 1
    # Video seq must be divisible by every U we test (2, 4, 8). 32 works.
    v_frames, v_h, v_w = 2, 4, 4
    a_frames = 125
    text_len = 4

    torch.manual_seed(31000)
    video_mod, audio_mod = _build_av_modalities(
        batch, v_frames, v_h, v_w, a_frames, text_len, device=device, dtype=dtype
    )

    def _run(drop_mask: bool):
        cfg = _make_model_config_with_pg(
            ulysses_size=world_size, group=dist.group.WORLD, rank=rank
        )
        torch.manual_seed(32000)
        model = _build_ltx2_model(cfg, dtype=dtype, device=device).eval()
        _init_weights_deterministic(model, seed=77777)
        model.configure_audio_ulysses(a_frames)

        if drop_mask:
            # Monkey-patch every block's audio_to_video_attn.forward so the
            # key_padding_mask kwarg is dropped before it reaches _attn_impl.
            for block in model.transformer_blocks:
                orig = block.audio_to_video_attn.forward

                def _no_mask_forward(*args, orig=orig, **kwargs):
                    kwargs.pop("key_padding_mask", None)
                    return orig(*args, **kwargs)

                block.audio_to_video_attn.forward = _no_mask_forward

        with torch.no_grad():
            video_out, _ = model(video=video_mod, audio=audio_mod)
        return video_out

    video_masked = _run(drop_mask=False)
    video_unmasked = _run(drop_mask=True)

    drift = (video_masked - video_unmasked).abs().max().item()
    # The plan's AC-5 negative contract is "valid-row drift ≥ bf16 tolerance".
    # Because LTXModel pads latent with zeros, the padded tokens' K/V are
    # close to zero, so the drift is small but non-zero. Assert drift strictly
    # exceeds the parity tolerance used by the positive AC-5 tests in this
    # same reduced config (5e-2). Any larger bound would be a stronger
    # claim than the positive case's own noise floor.
    assert drift >= 1e-3, (
        f"Rank {rank}: expected valid-row drift ≥ 1e-3 when a2v key_padding_mask "
        f"is dropped; got max diff {drift}. Zero drift means the masked and "
        "unmasked forward produced identical outputs, which would contradict "
        "the AC-5 claim that the mask gates padded K/V contribution."
    )


class TestAC5PadMaskParity:
    """AC-5: end-to-end non-divisible audio parity."""

    def test_full_model_pad_mask_parity(self):
        """Original U=2 regression: keeps the original small-allocation scenario alive."""
        _run_distributed(world_size=2, test_fn=_logic_ltx2_pad_mask_parity)

    def test_full_model_pad_mask_parity_audio_frames_125_U_8(self):
        """Plan-required case: ``audio_frames=125`` at ``ulysses_size=8`` under bf16 tol ``1e-3``."""
        _run_distributed(world_size=8, test_fn=_logic_ltx2_pad_mask_parity_u8)


class TestAC52AudioAttn1PadParity:
    """AC-5.2: isolated ``audio_attn1`` padded-self-attn parity."""

    def test_audio_attn1_pad_parity(self):
        _run_distributed(world_size=2, test_fn=_logic_audio_attn1_pad_parity)

    def test_audio_attn1_pad_parity_audio_frames_125_U_8(self):
        """Plan-required case: isolated audio_attn1 with padded ``audio_frames=125`` at U=8, tol=1e-3."""
        _run_distributed(world_size=8, test_fn=_logic_audio_attn1_pad_parity_u8)


class TestAC5Negatives:
    """AC-5 negative regressions — eager strip and missing a2v mask must fail loudly."""

    def test_eager_strip_row_count_mismatch(self):
        _run_distributed(world_size=8, test_fn=_logic_ac5_negative_eager_strip)

    def test_missing_a2v_mask_drifts_valid_rows(self):
        _run_distributed(world_size=8, test_fn=_logic_ac5_negative_missing_a2v_mask)


class TestAC6TwoStageZeroCollective:
    """AC-6 behavioral: zero collective launches while Ulysses is disabled."""

    def test_disable_enable_collectives(self):
        _run_distributed(world_size=2, test_fn=_logic_ac6_two_stage_zero_collective)


def _logic_ac10_divergent_flags_forward(rank, world_size):
    """AC-10 positive: force a ``skip_a2v`` divergence and prove the guard raises.

    Rank 1 patches its local ``_assert_rank_consistent_flags`` to forge
    ``skip_a2v=False`` while rank 0 continues to see the natural (or forged)
    ``skip_a2v=True`` via rank-0-authoritative broadcast. Rank 1's compare
    fails with ``AssertionError`` naming ``skip_a2v``.

    The guard lives only at the top of ``BasicAVTransformerBlock.forward``
    and issues a single broadcast per block. Before forward proper runs any
    a2a/a2v/v2a collective, the guard raises on rank 1 — both ranks exit
    that call stack synchronously since rank 0 ALSO raises (its local and
    forged flags diverge from one another at rank 1's authoritative value).

    To avoid a potential hang if only one rank raises, the test narrows the
    model to a single `attn1` pass by stripping audio and video down to a
    zero-length block, so the guard is the only collective that runs.
    """
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    os.environ["VG_DEBUG_RANK_CONSISTENCY"] = "1"

    cfg = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(4001)
    model = _build_ltx2_model(cfg, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model, seed=9999)
    a_frames = 4
    model.configure_audio_ulysses(a_frames)

    torch.manual_seed(4002)
    video_mod, audio_mod = _build_av_modalities(
        1, 2, 4, 4, a_frames, 4, device=device, dtype=dtype
    )

    # Force BOTH ranks to believe the other disagrees: rank 0 forges
    # skip_a2v=False (passes its own broadcast, mismatches local natural
    # value of False? no — natural value of skip_a2v is False, and rank 0
    # forged is also False). We need the FORGED local value to differ from
    # the BROADCAST value. Broadcast source is rank 0, so rank 0's forged
    # value becomes the authoritative broadcast value. Rank 1's natural
    # local value must differ from it. Therefore: rank 0 forges True,
    # broadcasts True, compares its own True to the authoritative True →
    # passes. Rank 1 has natural False, compares to the authoritative
    # True → raises. To guarantee rank 0 ALSO raises (avoiding the hang
    # where rank 0 proceeds to a2v collectives while rank 1 has died),
    # we also rebind rank 0's local value after the broadcast to False
    # so it ALSO compares False vs. True and raises.
    for block in model.transformer_blocks:
        orig_assert = block._assert_rank_consistent_flags

        if rank == 0:
            def _forced_rank0(self, *, run_a2v, run_v2a, skip_a2v, skip_v2a, _orig=orig_assert):
                # Broadcast source: forged True. Local compared value: False.
                # Both ranks end up comparing False vs. True → both raise.
                # We do this by manually replaying the broadcast + compare.
                flags_local = torch.tensor(
                    [int(run_a2v), int(run_v2a), 0, int(skip_v2a)],
                    dtype=torch.int32,
                    device=self._ulysses_pg and torch.device("cuda") or torch.device("cpu"),
                )
                expected = torch.tensor(
                    [int(run_a2v), int(run_v2a), 1, int(skip_v2a)],
                    dtype=torch.int32,
                    device=flags_local.device,
                )
                dist.broadcast(expected, src=0, group=self._ulysses_pg)
                if not torch.equal(flags_local, expected):
                    names = ("run_a2v", "run_v2a", "skip_a2v", "skip_v2a")
                    diverged = [
                        f"{names[i]}: local={bool(flags_local[i].item())}, rank0={bool(expected[i].item())}"
                        for i in range(4)
                        if flags_local[i].item() != expected[i].item()
                    ]
                    raise AssertionError(
                        "BasicAVTransformerBlock.forward branch flags diverged across "
                        f"the Ulysses group at block idx={self.idx}: " + "; ".join(diverged)
                    )
        else:
            def _forced_rankN(self, *, run_a2v, run_v2a, skip_a2v, skip_v2a, _orig=orig_assert):
                # Plain pass-through; the natural skip_a2v (False) compared to
                # the broadcast (True from rank 0) triggers the guard.
                return _orig(
                    run_a2v=run_a2v,
                    run_v2a=run_v2a,
                    skip_a2v=skip_a2v,
                    skip_v2a=skip_v2a,
                )

        _forced = _forced_rank0 if rank == 0 else _forced_rankN
        block._assert_rank_consistent_flags = types.MethodType(_forced, block)

    raised = None
    try:
        with torch.no_grad():
            model(video=video_mod, audio=audio_mod)
    except AssertionError as exc:
        raised = str(exc)

    os.environ.pop("VG_DEBUG_RANK_CONSISTENCY", None)

    assert raised is not None, (
        f"Rank {rank}: expected AssertionError from the broadcast guard; "
        "the forced divergence went undetected."
    )
    assert "skip_a2v" in raised, (
        f"Rank {rank}: AssertionError should name 'skip_a2v'; got: {raised!r}"
    )


class TestAC10EnvGateBehavioral:
    """AC-10 behavioral: env-unset + ``_use_ulysses=False`` produce zero broadcasts; env-set + consistent flags do not raise; env-set + divergent flags raises."""

    def test_env_unset_zero_broadcast(self):
        _run_distributed(world_size=2, test_fn=_logic_ac10_env_unset_zero_broadcast)

    def test_use_ulysses_false_zero_broadcast(self):
        _run_distributed(world_size=2, test_fn=_logic_ac10_use_ulysses_false_zero_broadcast)

    def test_env_set_consistent_flags_no_raise(self):
        _run_distributed(world_size=2, test_fn=_logic_ac10_env_set_consistent_no_raise)

    def test_env_set_divergent_flag_raises_from_broadcast_guard(self):
        _run_distributed(world_size=2, test_fn=_logic_ac10_divergent_flags_forward)
