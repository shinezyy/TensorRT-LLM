# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`UlyssesCrossAttention`.

Covers:
    - Single-rank (``world_size == 1``) fast path bit-exactness vs. direct inner
      backend and zero distributed-op launches.
    - ``support_fused_qkv`` surface.
    - ``all_to_all_5d`` round-trip correctness on the stacked ``[B, S/U, 2, H, D]``
      tensor shape (the fused K|V payload).
    - Multi-rank (``world_size > 1``) parity against a reference single-GPU
      computation, exercised via ``torch.multiprocessing.spawn`` with the
      gloo backend on CPU (no GPUs required for correctness).
"""

import math
import os
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

try:
    from tensorrt_llm._torch.distributed import all_to_all_4d, all_to_all_5d
    from tensorrt_llm._torch.visual_gen.attention_backend import (
        UlyssesCrossAttention,
        VanillaAttention,
    )
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not MODULES_AVAILABLE, reason="Required modules not available"
)


# ---------------------------------------------------------------------------
# Distributed test harness (gloo on CPU — correctness does not require GPUs).
# ---------------------------------------------------------------------------


def _init_distributed_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, test_fn, port):
    try:
        _init_distributed_worker(rank, world_size, port)
        test_fn(rank, world_size)
    finally:
        _cleanup_distributed()


def _run_distributed(world_size: int, test_fn: Callable) -> None:
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, test_fn, port),
        nprocs=world_size,
        join=True,
    )


# ---------------------------------------------------------------------------
# Module-level worker logic (must be picklable for mp.spawn).
# ---------------------------------------------------------------------------


def _logic_world_size_1_fast_path(rank, world_size):
    """``world_size == 1`` path is bit-exact with a direct inner backend call."""
    batch, h, h_kv, s_q, s_kv, head_dim = 1, 4, 2, 13, 17, 32

    inner = VanillaAttention(num_heads=h, num_kv_heads=h_kv, head_dim=head_dim)
    wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)
    assert wrapper.world_size == 1

    torch.manual_seed(42)
    q = torch.randn(batch, s_q, h, head_dim)
    k = torch.randn(batch, s_kv, h_kv, head_dim)
    v = torch.randn(batch, s_kv, h_kv, head_dim)

    out = wrapper.forward(q, k, v)

    # Direct call: wrapper transposes NHD->HND for VanillaAttention (HND preferred).
    direct = inner.forward(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    torch.testing.assert_close(out, direct, rtol=1e-5, atol=1e-5)


def _logic_a2a_5d_kv_stack_roundtrip(rank, world_size):
    """Round-trip a fused K|V stack (``dim=2`` size 2) through ``all_to_all_5d``."""
    batch = 2
    seq_per_rank = 4
    h_kv = world_size * 3
    head_dim = 16

    # [B, S/U, 2, H_kv, D]: fused K|V payload shape used by UlyssesCrossAttention.
    torch.manual_seed(100 + rank)
    k = torch.randn(batch, seq_per_rank, h_kv, head_dim)
    v = torch.randn(batch, seq_per_rank, h_kv, head_dim)
    kv_stack = torch.stack([k, v], dim=2)
    assert kv_stack.shape == (batch, seq_per_rank, 2, h_kv, head_dim)

    # Forward a2a: [B, S/U, 2, H_kv, D] -> [B, S, 2, H_kv/U, D].
    gathered = all_to_all_5d(
        kv_stack, scatter_dim=3, gather_dim=1, process_group=None
    )
    assert gathered.shape == (
        batch,
        seq_per_rank * world_size,
        2,
        h_kv // world_size,
        head_dim,
    )

    # Unbind along dim=2: should cleanly split into K and V on each rank.
    k_full, v_full = gathered.unbind(dim=2)
    assert k_full.shape == (
        batch,
        seq_per_rank * world_size,
        h_kv // world_size,
        head_dim,
    )
    assert v_full.shape == k_full.shape

    # Compare to per-tensor all_to_all_4d applied independently: must match bit-exact.
    k_ref = all_to_all_4d(k, scatter_dim=2, gather_dim=1, process_group=None)
    v_ref = all_to_all_4d(v, scatter_dim=2, gather_dim=1, process_group=None)
    torch.testing.assert_close(k_full.contiguous(), k_ref, rtol=0, atol=0)
    torch.testing.assert_close(v_full.contiguous(), v_ref, rtol=0, atol=0)

    # Round-trip: gather back should equal the original shards.
    gathered_back = all_to_all_5d(
        gathered, scatter_dim=1, gather_dim=3, process_group=None
    )
    torch.testing.assert_close(gathered_back, kv_stack, rtol=1e-5, atol=1e-5)


def _logic_ulysses_cross_parity_vs_single_gpu(rank, world_size):
    """Multi-rank ``UlyssesCrossAttention`` matches a single-device reference."""
    batch = 1
    u = world_size
    s_q = u * 7
    s_kv = u * 11
    h = u * 4
    h_kv = u * 2
    head_dim = 32
    scale = 1.0 / math.sqrt(head_dim)

    # Every rank generates the same full tensors with the same seed.
    torch.manual_seed(2026)
    q_full = torch.randn(batch, s_q, h, head_dim)
    k_full = torch.randn(batch, s_kv, h_kv, head_dim)
    v_full = torch.randn(batch, s_kv, h_kv, head_dim)

    # Each rank takes its seq shard.
    q_shard = q_full[:, rank * (s_q // u) : (rank + 1) * (s_q // u)].contiguous()
    k_shard = k_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()
    v_shard = v_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()

    # Inner backend is instantiated with per-rank head counts (H/U, H_kv/U).
    inner = VanillaAttention(
        num_heads=h // u, num_kv_heads=h_kv // u, head_dim=head_dim
    )
    wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)

    out_shard = wrapper.forward(q_shard, k_shard, v_shard)
    assert out_shard.shape == (batch, s_q // u, h, head_dim), (
        f"Rank {rank}: expected output shape {(batch, s_q // u, h, head_dim)}, "
        f"got {tuple(out_shard.shape)}"
    )

    # Reference: run SDPA on full tensors in HND layout.
    ref = F.scaled_dot_product_attention(
        q_full.transpose(1, 2),
        k_full.transpose(1, 2),
        v_full.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2).contiguous()
    expected_shard = ref[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]

    torch.testing.assert_close(
        out_shard, expected_shard, rtol=1e-4, atol=1e-4
    )


def _logic_ulysses_cross_forwards_key_padding_mask(rank, world_size):
    """``key_padding_mask`` threads through the wrapper into the inner ``VanillaAttention``."""
    batch = 1
    u = world_size
    s_q = u * 4
    s_kv = u * 6
    h = u * 2
    h_kv = u * 2
    head_dim = 16
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(7)
    q_full = torch.randn(batch, s_q, h, head_dim)
    k_full = torch.randn(batch, s_kv, h_kv, head_dim)
    v_full = torch.randn(batch, s_kv, h_kv, head_dim)

    # Pad the last position on the KV side.
    pad_mask = torch.ones(batch, s_kv, dtype=torch.bool)
    pad_mask[:, -u:] = False  # Pad the full final shard-slot across all ranks.

    q_shard = q_full[:, rank * (s_q // u) : (rank + 1) * (s_q // u)].contiguous()
    k_shard = k_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()
    v_shard = v_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()

    inner = VanillaAttention(
        num_heads=h // u, num_kv_heads=h_kv // u, head_dim=head_dim
    )
    wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)
    out_shard = wrapper.forward(q_shard, k_shard, v_shard, key_padding_mask=pad_mask)

    # Reference: unpadded-slice SDPA on full tensors.
    s_valid = s_kv - u
    ref = F.scaled_dot_product_attention(
        q_full.transpose(1, 2),
        k_full[:, :s_valid].transpose(1, 2),
        v_full[:, :s_valid].transpose(1, 2),
        scale=scale,
    ).transpose(1, 2).contiguous()
    expected_shard = ref[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]

    torch.testing.assert_close(out_shard, expected_shard, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUlyssesCrossAttentionSingleRank:
    """Behaviors that do not require multi-process spawn."""

    def test_support_fused_qkv_is_false(self):
        """``UlyssesCrossAttention`` cannot fuse Q with K/V because ``S_q != S_kv``."""
        assert UlyssesCrossAttention.support_fused_qkv() is False

    def test_world_size_1_fast_path_matches_direct_call(self):
        """No-distributed fast path is bit-exact with a direct inner backend call."""
        batch, h, h_kv, s_q, s_kv, head_dim = 1, 4, 2, 13, 17, 32

        inner = VanillaAttention(num_heads=h, num_kv_heads=h_kv, head_dim=head_dim)
        wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)
        assert wrapper.world_size == 1
        # Head counts passthrough: with world_size=1, the "full" heads equal the inner's.
        assert wrapper.num_heads == h
        assert wrapper.num_kv_heads == h_kv

        torch.manual_seed(42)
        q = torch.randn(batch, s_q, h, head_dim)
        k = torch.randn(batch, s_kv, h_kv, head_dim)
        v = torch.randn(batch, s_kv, h_kv, head_dim)

        out = wrapper.forward(q, k, v)

        direct = inner.forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

        torch.testing.assert_close(out, direct, rtol=1e-5, atol=1e-5)
        assert out.shape == (batch, s_q, h, head_dim)

    def test_world_size_1_fast_path_launches_no_collectives(self):
        """Fast path calls no ``torch.distributed`` collective; no process group is touched."""
        batch, h, h_kv, s_q, s_kv, head_dim = 1, 2, 2, 5, 7, 16

        inner = VanillaAttention(num_heads=h, num_kv_heads=h_kv, head_dim=head_dim)
        wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)

        q = torch.randn(batch, s_q, h, head_dim)
        k = torch.randn(batch, s_kv, h_kv, head_dim)
        v = torch.randn(batch, s_kv, h_kv, head_dim)

        # Instrument: patch the a2a primitives to a tripwire. If the fast path
        # hits them even once, the test fails. Importing here so the patch is
        # scoped to this test only.
        from tensorrt_llm._torch.visual_gen.attention_backend import parallel as _parallel

        launched = {"count": 0}

        def _tripwire(*_args, **_kwargs):
            launched["count"] += 1
            raise AssertionError("world_size==1 fast path must not launch collectives")

        orig_4d = _parallel.all_to_all_4d
        orig_5d = _parallel.all_to_all_5d
        _parallel.all_to_all_4d = _tripwire
        _parallel.all_to_all_5d = _tripwire
        try:
            wrapper.forward(q, k, v)
        finally:
            _parallel.all_to_all_4d = orig_4d
            _parallel.all_to_all_5d = orig_5d

        assert launched["count"] == 0

    def test_shape_assertions(self):
        """Runtime shape assertions fire on malformed inputs."""
        h, h_kv, head_dim = 2, 2, 16
        inner = VanillaAttention(num_heads=h, num_kv_heads=h_kv, head_dim=head_dim)
        wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)

        q = torch.randn(1, 4, h, head_dim)
        k = torch.randn(1, 6, h_kv, head_dim)
        v_wrong_batch = torch.randn(2, 6, h_kv, head_dim)
        with pytest.raises(AssertionError, match="batch mismatch"):
            wrapper.forward(q, k, v_wrong_batch)

        k_wrong_seq = torch.randn(1, 5, h_kv, head_dim)
        v_seq_6 = torch.randn(1, 6, h_kv, head_dim)
        with pytest.raises(AssertionError, match="seq shard mismatch"):
            wrapper.forward(q, k_wrong_seq, v_seq_6)

        q_3d = torch.randn(1, 4, head_dim)
        v_ok = torch.randn(1, 6, h_kv, head_dim)
        with pytest.raises(AssertionError, match="q must be 4D"):
            wrapper.forward(q_3d, k, v_ok)


class TestUlyssesCrossAttentionWorldSize1ViaSpawn:
    """Run the fast path inside a ``world_size=1`` spawned worker for parity with multi-rank tests."""

    def test_world_size_1_spawned(self):
        _run_distributed(world_size=1, test_fn=_logic_world_size_1_fast_path)

    def test_all_to_all_5d_kv_stack_roundtrip_world_size_1(self):
        _run_distributed(world_size=1, test_fn=_logic_a2a_5d_kv_stack_roundtrip)


class TestUlyssesCrossAttentionMultiRank:
    """Multi-process tests using gloo on CPU (no GPUs required)."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_all_to_all_5d_kv_stack_roundtrip(self, world_size):
        """Fused K|V stack round-trip matches per-tensor ``all_to_all_4d`` bit-exact."""
        _run_distributed(world_size=world_size, test_fn=_logic_a2a_5d_kv_stack_roundtrip)

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_ulysses_cross_parity_vs_single_gpu(self, world_size):
        """Multi-rank forward matches the single-device SDPA reference on each rank's output shard."""
        _run_distributed(
            world_size=world_size,
            test_fn=_logic_ulysses_cross_parity_vs_single_gpu,
        )

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_key_padding_mask_forwards_through_wrapper(self, world_size):
        """``key_padding_mask`` reaches the inner backend via ``**kwargs`` unchanged."""
        _run_distributed(
            world_size=world_size,
            test_fn=_logic_ulysses_cross_forwards_key_padding_mask,
        )
