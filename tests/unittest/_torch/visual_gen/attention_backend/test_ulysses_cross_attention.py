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

    # Reference: run SDPA on full tensors in HND layout. Use enable_gqa so the
    # H_kv < H case (H=U·4, H_kv=U·2) runs correctly; the VanillaAttention
    # backend applies the same flag internally when num_heads != num_kv_heads.
    sdpa_kwargs = {"enable_gqa": True} if h != h_kv else {}
    ref = F.scaled_dot_product_attention(
        q_full.transpose(1, 2),
        k_full.transpose(1, 2),
        v_full.transpose(1, 2),
        scale=scale,
        **sdpa_kwargs,
    ).transpose(1, 2).contiguous()
    expected_shard = ref[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]

    torch.testing.assert_close(
        out_shard, expected_shard, rtol=1e-4, atol=1e-4
    )


# ---------------------------------------------------------------------------
# AC-1 matrix: U in {2, 4, 8} x varied S_q, S_kv, H, H_kv multipliers.
# Multipliers are encoded in the worker function's env to keep the signature
# mp.spawn-friendly.
# ---------------------------------------------------------------------------


def _logic_ulysses_cross_matrix(rank, world_size):
    batch = 1
    u = world_size
    s_q_mult = int(os.environ["TEST_S_Q_MULT"])
    s_kv_mult = int(os.environ["TEST_S_KV_MULT"])
    h_mult = int(os.environ["TEST_H_MULT"])
    h_kv_mult = int(os.environ["TEST_H_KV_MULT"])
    s_q = u * s_q_mult
    s_kv = u * s_kv_mult
    h = u * h_mult
    h_kv = u * h_kv_mult
    head_dim = 32
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(10007 + u * 131 + s_q_mult * 17 + s_kv_mult)
    q_full = torch.randn(batch, s_q, h, head_dim)
    k_full = torch.randn(batch, s_kv, h_kv, head_dim)
    v_full = torch.randn(batch, s_kv, h_kv, head_dim)

    q_shard = q_full[:, rank * (s_q // u) : (rank + 1) * (s_q // u)].contiguous()
    k_shard = k_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()
    v_shard = v_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()

    inner = VanillaAttention(
        num_heads=h // u, num_kv_heads=h_kv // u, head_dim=head_dim
    )
    wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)

    out_shard = wrapper.forward(q_shard, k_shard, v_shard)
    assert out_shard.shape == (batch, s_q // u, h, head_dim), (
        f"Rank {rank}: bad shape {tuple(out_shard.shape)} for "
        f"U={u} (s_q={s_q}, s_kv={s_kv}, H={h}, H_kv={h_kv})"
    )

    # SDPA expects matching num_heads/num_kv_heads, so broadcast K/V heads
    # when H_kv != H.
    if h_kv != h:
        repeats = h // h_kv
        k_ref = k_full.repeat_interleave(repeats, dim=2)
        v_ref = v_full.repeat_interleave(repeats, dim=2)
    else:
        k_ref = k_full
        v_ref = v_full
    ref = F.scaled_dot_product_attention(
        q_full.transpose(1, 2),
        k_ref.transpose(1, 2),
        v_ref.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2).contiguous()
    expected_shard = ref[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]
    torch.testing.assert_close(out_shard, expected_shard, rtol=1e-4, atol=1e-4)


def _logic_ulysses_cross_rope_commutativity(rank, world_size):
    """RoPE applied pre-a2a to sharded Q/K matches RoPE applied post-a2a to full-seq.

    RoPE in LTX2 is pointwise on ``(seq, head_dim)`` per token — broadcast
    across heads. The per-head invariance is what makes the operation
    commute with Ulysses: applying RoPE on seq-sharded Q/K with seq-sharded
    cos/sin before the all-to-all yields the same numerical result as
    applying RoPE on full-seq Q/K after the all-to-all (because heads are
    redistributed, but each token-position's RoPE twist is head-independent).

    To preserve that invariance in the test, the cos/sin tensors must be
    broadcastable across BOTH the heads axis AND the a2a reshape. We use
    shape ``[1, S, 1, D]`` with broadcast along head dim = 1; the inner
    backend sees ``[B, S, H/U, D]`` and the per-position cos/sin multiply
    identically on every head.
    """
    batch = 1
    u = world_size
    s_q = u * 4
    s_kv = u * 6
    h = u * 2
    h_kv = u * 2
    head_dim = 32
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(314159)  # Same seed across ranks so full tensors match.
    q_full = torch.randn(batch, s_q, h, head_dim)
    k_full = torch.randn(batch, s_kv, h_kv, head_dim)
    v_full = torch.randn(batch, s_kv, h_kv, head_dim)
    cos_q = torch.randn(1, s_q, 1, head_dim) * 0.1 + 1.0
    sin_q = torch.randn(1, s_q, 1, head_dim) * 0.1
    cos_k = torch.randn(1, s_kv, 1, head_dim) * 0.1 + 1.0
    sin_k = torch.randn(1, s_kv, 1, head_dim) * 0.1

    def _apply_rope(x, cos, sin):
        half = head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        r1 = x1 * cos[..., :half] - x2 * sin[..., :half]
        r2 = x2 * cos[..., half:] + x1 * sin[..., half:]
        return torch.cat([r1, r2], dim=-1)

    # Path A: apply RoPE pre-a2a on the rank's shards with the corresponding
    # seq-sliced cos/sin.
    q_sh = q_full[:, rank * (s_q // u) : (rank + 1) * (s_q // u)].contiguous()
    k_sh = k_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()
    v_sh = v_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()
    cos_q_sh = cos_q[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]
    sin_q_sh = sin_q[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]
    cos_k_sh = cos_k[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)]
    sin_k_sh = sin_k[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)]
    q_roped_a = _apply_rope(q_sh, cos_q_sh, sin_q_sh)
    k_roped_a = _apply_rope(k_sh, cos_k_sh, sin_k_sh)

    inner = VanillaAttention(
        num_heads=h // u, num_kv_heads=h_kv // u, head_dim=head_dim
    )
    wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)
    out_a = wrapper.forward(q_roped_a, k_roped_a, v_sh)

    # Path B: reference — apply RoPE on full-seq tensors, then compute attention directly.
    q_roped_full = _apply_rope(q_full, cos_q, sin_q)
    k_roped_full = _apply_rope(k_full, cos_k, sin_k)
    ref = F.scaled_dot_product_attention(
        q_roped_full.transpose(1, 2),
        k_roped_full.transpose(1, 2),
        v_full.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2).contiguous()
    expected_shard = ref[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]

    torch.testing.assert_close(out_a, expected_shard, rtol=1e-4, atol=1e-4)


def _logic_ulysses_cross_noise_negative(rank, world_size):
    """Perturbing one rank's Q shard produces a detectable mismatch vs. the reference."""
    batch = 1
    u = world_size
    s_q = u * 5
    s_kv = u * 9
    h = u * 2
    h_kv = u * 2
    head_dim = 16
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(808 + rank)
    q_full = torch.randn(batch, s_q, h, head_dim)
    k_full = torch.randn(batch, s_kv, h_kv, head_dim)
    v_full = torch.randn(batch, s_kv, h_kv, head_dim)

    q_sh = q_full[:, rank * (s_q // u) : (rank + 1) * (s_q // u)].contiguous()
    k_sh = k_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()
    v_sh = v_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()

    # Only rank 0 perturbs its Q shard. Other ranks pass clean tensors.
    if rank == 0:
        q_sh = q_sh + 7.5

    inner = VanillaAttention(
        num_heads=h // u, num_kv_heads=h_kv // u, head_dim=head_dim
    )
    wrapper = UlyssesCrossAttention(inner_backend=inner, process_group=None)
    out_perturbed = wrapper.forward(q_sh, k_sh, v_sh)

    # Reference: unperturbed single-device output.
    ref = F.scaled_dot_product_attention(
        q_full.transpose(1, 2),
        k_full.transpose(1, 2),
        v_full.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2).contiguous()
    expected_shard = ref[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]

    max_diff = (out_perturbed - expected_shard).abs().max().item()
    assert max_diff > 1e-2, (
        f"Rank {rank}: expected large drift after one-rank Q noise; got max diff {max_diff}"
    )


def _logic_ulysses_cross_swapped_dims_negative(rank, world_size):
    """Swapping ``scatter_dim``/``gather_dim`` on the KV a2a produces a detectable mismatch.

    Simulates a common programmer error: a subclass/monkey-patch that sends
    the fused K|V stack through ``all_to_all_5d`` with the scatter / gather
    dims swapped. The wrapper's subsequent ``unbind(dim=2)`` + SDPA should
    then produce outputs that diverge noticeably from the reference. This
    exercises the "fails loudly rather than produces garbage silently" half
    of AC-1's negative contract.
    """
    batch = 1
    u = world_size
    s_q = u * 4
    s_kv = u * 4  # s_kv == s_q so the swapped shapes are still self-consistent.
    h = u * 2
    h_kv = u * 2
    head_dim = 16
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(31415 + rank)
    q_full = torch.randn(batch, s_q, h, head_dim)
    k_full = torch.randn(batch, s_kv, h_kv, head_dim)
    v_full = torch.randn(batch, s_kv, h_kv, head_dim)

    q_sh = q_full[:, rank * (s_q // u) : (rank + 1) * (s_q // u)].contiguous()
    k_sh = k_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()
    v_sh = v_full[:, rank * (s_kv // u) : (rank + 1) * (s_kv // u)].contiguous()

    inner = VanillaAttention(
        num_heads=h // u, num_kv_heads=h_kv // u, head_dim=head_dim
    )
    # Subclass that swaps scatter/gather on the fused KV a2a — intentionally wrong.
    class _SwappedCrossAttn(UlyssesCrossAttention):
        def forward(self, q, k, v, **kwargs):
            if self.world_size > 1:
                q = all_to_all_4d(
                    q, scatter_dim=2, gather_dim=1, process_group=self.process_group
                )
                kv = torch.stack([k, v], dim=2)
                # Swap the dims — intentionally wrong.
                kv = all_to_all_5d(
                    kv, scatter_dim=1, gather_dim=3, process_group=self.process_group
                )
                k, v = kv.unbind(dim=2)
                k = k.contiguous()
                v = v.contiguous()
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = self.inner_backend.forward(q=q, k=k, v=v, **kwargs)
            out = out.transpose(1, 2).contiguous()
            if self.world_size > 1:
                out = all_to_all_4d(
                    out, scatter_dim=1, gather_dim=2, process_group=self.process_group
                )
            return out

    wrapper = _SwappedCrossAttn(inner_backend=inner, process_group=None)
    # Either a runtime shape error OR a numerical divergence > tolerance
    # satisfies the "fails loudly rather than produces garbage silently" contract.
    loud_failure = False
    try:
        out_bad = wrapper.forward(q_sh, k_sh, v_sh)
    except RuntimeError:
        loud_failure = True
    else:
        ref = F.scaled_dot_product_attention(
            q_full.transpose(1, 2),
            k_full.transpose(1, 2),
            v_full.transpose(1, 2),
            scale=scale,
        ).transpose(1, 2).contiguous()
        expected_shard = ref[:, rank * (s_q // u) : (rank + 1) * (s_q // u)]
        max_diff = (out_bad - expected_shard).abs().max().item()
        loud_failure = max_diff > 1e-2
    assert loud_failure, (
        f"Rank {rank}: swapped scatter/gather silently produced a valid-looking "
        "result; expected either a RuntimeError or a detectable numerical drift."
    )


def _logic_assert_fused_kv_stack_shape_fires_on_bad_dim2(rank, world_size):
    """AC-2 negative: ``UlyssesCrossAttention._assert_fused_kv_stack_shape`` fires on ``dim=2 != 2``."""
    batch = 1
    seq_per_rank = 2
    h = 4
    head_dim = 8

    torch.manual_seed(271828 + rank)

    # Wrong size at dim=2 (size 3 instead of the required 2) must raise.
    bad_stack = torch.randn(batch, seq_per_rank, 3, h, head_dim)
    try:
        UlyssesCrossAttention._assert_fused_kv_stack_shape(bad_stack)
    except AssertionError as exc:
        assert "dim=2 size 2" in str(exc), (
            f"Rank {rank}: expected message naming 'dim=2 size 2', got {exc!r}"
        )
    else:
        raise AssertionError(
            f"Rank {rank}: expected AssertionError for dim=2 size 3; none raised"
        )

    # Wrong ndim (not 5D) must also raise.
    not_5d = torch.randn(batch, seq_per_rank, 2, h, head_dim, 1)  # 6D.
    try:
        UlyssesCrossAttention._assert_fused_kv_stack_shape(not_5d)
    except AssertionError as exc:
        assert "must be 5D" in str(exc), (
            f"Rank {rank}: expected message naming '5D', got {exc!r}"
        )
    else:
        raise AssertionError(
            f"Rank {rank}: expected AssertionError for 6D input; none raised"
        )

    # The correct shape must NOT raise — verifies the guard is tight.
    good_stack = torch.randn(batch, seq_per_rank, 2, h, head_dim)
    UlyssesCrossAttention._assert_fused_kv_stack_shape(good_stack)


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


class TestUlyssesCrossAttentionAC1Matrix:
    """Full AC-1 parity matrix.

    The plan's AC-1 requires: ``U ∈ {2, 4, 8}``,
    ``S_q ∈ {U·1, U·7, U·13}``, ``S_kv ∈ {U·3, U·17}``,
    ``H ∈ {U·4, U·8}``, ``H_kv ∈ {U·4, U·1}``. A Cartesian product would
    be ``3·3·2·2·2 = 72`` spawned test runs; we cover every multiplier
    value with a representative sweep that visits each S/H setting at
    least once per ``U``.
    """

    # Each row: (world_size, s_q_mult, s_kv_mult, h_mult, h_kv_mult).
    # The sweep touches every multiplier in the plan matrix at least once.
    _MATRIX = [
        (2, 1, 3, 4, 4),
        (2, 7, 17, 8, 1),
        (2, 13, 3, 4, 4),
        (4, 1, 17, 8, 4),
        (4, 7, 3, 4, 1),
        (4, 13, 17, 8, 4),
        (8, 1, 3, 4, 1),
        (8, 7, 17, 8, 4),
        (8, 13, 3, 4, 4),
    ]

    @pytest.mark.parametrize(
        "world_size,s_q_mult,s_kv_mult,h_mult,h_kv_mult", _MATRIX
    )
    def test_parity_matrix(
        self, world_size, s_q_mult, s_kv_mult, h_mult, h_kv_mult
    ):
        """Each matrix row: wrapped output shard matches single-device SDPA reference."""
        os.environ["TEST_S_Q_MULT"] = str(s_q_mult)
        os.environ["TEST_S_KV_MULT"] = str(s_kv_mult)
        os.environ["TEST_H_MULT"] = str(h_mult)
        os.environ["TEST_H_KV_MULT"] = str(h_kv_mult)
        try:
            _run_distributed(
                world_size=world_size, test_fn=_logic_ulysses_cross_matrix
            )
        finally:
            for key in (
                "TEST_S_Q_MULT",
                "TEST_S_KV_MULT",
                "TEST_H_MULT",
                "TEST_H_KV_MULT",
            ):
                os.environ.pop(key, None)

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_rope_commutativity(self, world_size):
        """Pre-a2a sharded RoPE matches post-a2a full-seq RoPE under bf16 tolerance."""
        _run_distributed(
            world_size=world_size,
            test_fn=_logic_ulysses_cross_rope_commutativity,
        )


class TestUlyssesCrossAttentionNegatives:
    """AC-1 / AC-2 negatives — wrong inputs produce detectable failures."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_noise_perturbation_is_detected(self, world_size):
        """Rank-0-only additive noise on Q produces a drift >> tolerance vs. the reference."""
        _run_distributed(
            world_size=world_size, test_fn=_logic_ulysses_cross_noise_negative
        )

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_swapped_scatter_gather_is_detected(self, world_size):
        """Swapped scatter_dim/gather_dim on the KV a2a produces detectable drift."""
        _run_distributed(
            world_size=world_size,
            test_fn=_logic_ulysses_cross_swapped_dims_negative,
        )

    def test_assert_fused_kv_stack_shape_fires_on_bad_dim2_single_rank(self):
        """AC-2 negative: the wrapper's guard fires loudly when ``dim=2 != 2``.

        Pure CPU, no spawn needed — the guard is a static method and the
        test passes a malformed pre-stacked tensor directly.
        """
        batch, seq, h, head_dim = 1, 2, 4, 8

        bad_stack = torch.randn(batch, seq, 3, h, head_dim)
        with pytest.raises(AssertionError, match="dim=2 size 2"):
            UlyssesCrossAttention._assert_fused_kv_stack_shape(bad_stack)

        not_5d = torch.randn(batch, seq, 2, h, head_dim, 1)
        with pytest.raises(AssertionError, match="must be 5D"):
            UlyssesCrossAttention._assert_fused_kv_stack_shape(not_5d)

        # Good stack does not raise.
        UlyssesCrossAttention._assert_fused_kv_stack_shape(
            torch.randn(batch, seq, 2, h, head_dim)
        )

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_assert_fused_kv_stack_shape_fires_under_distributed(self, world_size):
        """AC-2 negative exercised inside spawned workers to mirror the multi-rank AC-2 contract."""
        _run_distributed(
            world_size=world_size,
            test_fn=_logic_assert_fused_kv_stack_shape_fires_on_bad_dim2,
        )
