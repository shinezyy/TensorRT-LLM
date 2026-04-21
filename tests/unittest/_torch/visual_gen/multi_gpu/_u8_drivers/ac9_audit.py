# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AC-9 nsys audit: kernel counts inside AV cross-attn NVTX ranges.

The audit window is the two AV-direction cross-attention NVTX ranges
``ltx2.a2v_cross_attn`` and ``ltx2.v2a_cross_attn`` — text cross-attn
ranges (``*_cross_attn`` on the text stream) are intentionally excluded.

The HARD AC-9 contract checked here is **implementation-independent**:

  * Zero ``ncclDevKernel_AllGather*`` kernels inside any AV cross-attn
    range. This is the strict-Ulysses claim: the AllGather path has
    been removed from the AV cross-attn critical path.
  * Non-zero ``ncclDevKernel_AllToAll*`` / ``ncclDevKernel_SendRecv``
    kernels across the profile, so the replacement all-to-all path
    actually launches.

The plan's "exactly 3 AllToAll per range" wording is satisfied at the
OP level (Q a2a + fused K|V 5D a2a + output a2a), not at the NCCL kernel
level. On intra-node B200 topologies NCCL lowers
``dist.all_to_all_single`` to ``(U-1)`` pairwise ``ncclDevKernel_SendRecv``
kernels per op, and kernels can spill slightly outside the enclosing
Python NVTX range due to async launch queueing. An earlier version of
this script enforced ``a2a == 3`` per range; that check was removed
because it is topology-dependent and already disagrees with the
committed round-4 exports (per-range counts in the range 1–5). Per-range
raw counts are still reported in the JSON for human review.

Usage::

    python ac9_audit.py <profile>.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


_A2A_NVTX_NAMES = ("ltx2.a2v_cross_attn", "ltx2.v2a_cross_attn")

_OP_LEVEL_A2A_NVTX_NAMES = (
    "ulysses.cross.a2a.q",
    "ulysses.cross.a2a.kv",
    "ulysses.cross.a2a.out",
)


def _fetch_ranges(conn, names):
    # NVTX ranges use two encodings: registered strings (textId ->
    # StringIds.value) and inline strings (text column populated
    # directly). ``torch.cuda.nvtx.range`` uses the latter while the
    # ``ltx2.*`` markers go through the former. Union both.
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        f"""
        SELECT r.start, r.end, s.value
        FROM NVTX_EVENTS r JOIN StringIds s ON r.textId = s.id
        WHERE s.value IN ({placeholders})
        UNION ALL
        SELECT start, end, text
        FROM NVTX_EVENTS
        WHERE text IN ({placeholders})
        """,
        tuple(names) + tuple(names),
    ).fetchall()
    return [(int(start), int(end), str(value)) for (start, end, value) in rows]


def _fetch_kernel_events(conn):
    rows = conn.execute(
        """
        SELECT k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        """
    ).fetchall()
    return [(int(start), int(end), str(name)) for (start, end, name) in rows]


def _count_in_range(kernels, range_start, range_end, patterns):
    """Return total kernel count for any pattern in ``patterns`` whose kernel
    lives inside the ``[range_start, range_end]`` window."""
    if isinstance(patterns, str):
        patterns = (patterns,)
    total = 0
    for start, end, name in kernels:
        if end < range_start or start > range_end:
            continue
        if any(p in name for p in patterns):
            total += 1
    return total


def _theoretical_a2a_bytes_per_rank(
    *, U: int, B: int, S_kv: int, H_kv: int, D_h: int, elem_size: int
) -> int:
    """Plan's theoretical a2a byte count per K (or per V) per rank.

    From the plan §AC-9: each rank sends / receives
    ``((U-1)/U^2) * B * S_kv * H_kv * D_h * elem_size`` bytes per K (and
    per V) across all peers in one Q a2a / one fused K|V a2a / one
    output a2a. The audit reports the reference value so a post-hoc
    byte-aggregation pass can compare recorded NCCL payload bytes to
    this number at the ±10% tolerance the plan specifies.
    """
    return ((U - 1) * B * S_kv * H_kv * D_h * elem_size) // (U * U)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument(
        "--theoretical-bytes",
        nargs=6,
        metavar=("U", "B", "S_kv", "H_kv", "D_h", "elem_size"),
        type=int,
        default=None,
        help=(
            "If provided, emit the plan's per-rank, per-K (and per-V) "
            "a2a theoretical byte count under _totals.theoretical_bytes "
            "for external comparison against measured NCCL payloads."
        ),
    )
    args = parser.parse_args()

    if not args.sqlite.exists():
        print(f"ERROR: {args.sqlite} does not exist.", file=sys.stderr)
        sys.exit(2)

    conn = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)

    ranges = _fetch_ranges(conn, _A2A_NVTX_NAMES)
    if not ranges:
        print(f"ERROR: no 'ltx2.*cross_attn' NVTX ranges found in {args.sqlite}.",
              file=sys.stderr)
        sys.exit(3)
    kernels = _fetch_kernel_events(conn)
    # Op-level a2a NVTX ranges (Q / fused K|V / output), inserted inside
    # ``UlyssesCrossAttention.forward``. These are implementation-
    # independent: regardless of whether NCCL lowers ``all_to_all_single``
    # to a single ``AllToAll`` kernel or ``U-1`` ``SendRecv`` kernels per
    # op, there must be exactly one op-level NVTX range per a2a OP.
    op_ranges = _fetch_ranges(conn, _OP_LEVEL_A2A_NVTX_NAMES)

    summary = {}
    ok = True
    ALLTOALL_KERNEL_PATTERNS = (
        "ncclDevKernel_AllToAll",
        "ncclDevKernel_SendRecv",
    )
    ALLGATHER_KERNEL_PATTERNS = ("ncclDevKernel_AllGather",)
    for start, end, name in ranges:
        a2a = _count_in_range(kernels, start, end, ALLTOALL_KERNEL_PATTERNS)
        ag = _count_in_range(kernels, start, end, ALLGATHER_KERNEL_PATTERNS)
        # Count op-level NVTX ranges whose start time lies inside this
        # [range_start, range_end] window. The plan requires exactly 3:
        # Q a2a, fused K|V a2a, output a2a.
        op_count = sum(
            1 for (o_start, o_end, _o_name) in op_ranges
            if start <= o_start <= end
        )
        summary.setdefault(name, []).append({
            "all_to_all_kernels": a2a,
            "all_gather_kernels": ag,
            "op_level_a2a_nvtx": op_count,
        })
        if ag > 0:
            ok = False
            print(
                f"FAIL range={name} [{start},{end}]: AllGather kernels={ag} "
                "(plan requires 0 — Ulysses cross-attn must never launch AllGather).",
                file=sys.stderr,
            )
        # Op-level NVTX invariant: if op markers are present at all in this
        # profile, every AV cross-attn range must carry exactly 3 of them.
        # (If the profile was captured from a build without the markers
        # yet, ``op_ranges`` is empty and this check is skipped.)
        if op_ranges and op_count != 3:
            ok = False
            print(
                f"FAIL range={name} [{start},{end}]: op-level a2a NVTX "
                f"ranges={op_count} (expected exactly 3: Q / K|V / output).",
                file=sys.stderr,
            )

    total_a2a = sum(r["all_to_all_kernels"] for v in summary.values() for r in v)
    total_ag = sum(r["all_gather_kernels"] for v in summary.values() for r in v)
    total_op = sum(r["op_level_a2a_nvtx"] for v in summary.values() for r in v)
    summary["_totals"] = {
        "all_to_all_kernels": total_a2a,
        "all_gather_kernels": total_ag,
        "op_level_a2a_nvtx": total_op,
        "op_level_markers_present": bool(op_ranges),
        "num_ranges": sum(len(v) for v in summary.values()),
    }
    if args.theoretical_bytes is not None:
        U, B, S_kv, H_kv, D_h, elem_size = args.theoretical_bytes
        summary["_totals"]["theoretical_bytes_per_rank_per_tensor"] = (
            _theoretical_a2a_bytes_per_rank(
                U=U, B=B, S_kv=S_kv, H_kv=H_kv, D_h=D_h, elem_size=elem_size,
            )
        )
        summary["_totals"]["theoretical_bytes_params"] = {
            "U": U, "B": B, "S_kv": S_kv, "H_kv": H_kv,
            "D_h": D_h, "elem_size": elem_size,
        }
    if total_a2a == 0:
        ok = False
        print(
            "FAIL: no AllToAll-family kernels found across any AV cross-attn range "
            "(strict-Ulysses must launch NCCL SendRecv / AllToAll kernels).",
            file=sys.stderr,
        )

    print(json.dumps(summary, indent=2))
    if ok:
        print(
            "AC-9 PASS: every ltx2.{a2v,v2a}_cross_attn NVTX range has "
            "ZERO AllGather kernels, and AllToAll/SendRecv kernels are "
            "present globally. Per-range raw counts are reported above "
            "and are topology-dependent (intra-node NCCL compiles "
            "all_to_all_single to (U-1) SendRecv pairs per op, with "
            "async spillover outside the Python range)."
        )
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
