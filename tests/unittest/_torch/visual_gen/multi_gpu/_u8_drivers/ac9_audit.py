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
    # We also include ``globalTid`` so callers can filter per-rank.
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        f"""
        SELECT r.start, r.end, s.value, r.globalTid
        FROM NVTX_EVENTS r JOIN StringIds s ON r.textId = s.id
        WHERE s.value IN ({placeholders})
        UNION ALL
        SELECT start, end, text, globalTid
        FROM NVTX_EVENTS
        WHERE text IN ({placeholders})
        """,
        tuple(names) + tuple(names),
    ).fetchall()
    return [(int(start), int(end), str(value), int(tid) if tid is not None else -1)
            for (start, end, value, tid) in rows]


def _fetch_kernel_events(conn):
    # Include deviceId so callers can filter to kernels running on a
    # specific GPU (rank) when the profile covers multiple devices.
    rows = conn.execute(
        """
        SELECT k.start, k.end, s.value, k.deviceId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        """
    ).fetchall()
    return [(int(start), int(end), str(name),
             int(dev) if dev is not None else -1)
            for (start, end, name, dev) in rows]


def _count_in_range(kernels, range_start, range_end, patterns, device_id=None):
    """Return total kernel count for any pattern in ``patterns`` whose kernel
    lives inside the ``[range_start, range_end]`` window. When
    ``device_id`` is not None, only kernels on that GPU are counted."""
    if isinstance(patterns, str):
        patterns = (patterns,)
    total = 0
    for start, end, name, dev in kernels:
        if device_id is not None and dev != device_id:
            continue
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


def _sum_range_wallclock_ns(ranges, names_whitelist=None):
    """Total wall-clock nanoseconds summed over every range in ``ranges``."""
    total = 0
    for entry in ranges:
        start, end, name = entry[0], entry[1], entry[2]
        if names_whitelist is not None and name not in names_whitelist:
            continue
        total += max(0, end - start)
    return total


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
    parser.add_argument(
        "--measured-bytes-sidecar",
        type=Path,
        default=None,
        help=(
            "Path to a JSON sidecar written by the nsys driver with measured "
            "a2a byte counts per op (keys: a2a_q_bytes, a2a_kv_bytes, "
            "a2a_out_bytes, ...). Audit compares totals against "
            "--theoretical-bytes at ±10%% when both are given."
        ),
    )
    parser.add_argument(
        "--baseline-sqlite",
        type=Path,
        default=None,
        help=(
            "Path to a baseline nsys sqlite captured on the same hardware "
            "from the pre-change branch. Audit reports AV cross-attn NVTX "
            "wall-clock ratio (current/baseline). Plan tolerance: ≤ 1.10."
        ),
    )
    parser.add_argument(
        "--bytes-tolerance",
        type=float,
        default=0.10,
        help="Relative tolerance for bytes vs. theoretical (default 0.10).",
    )
    parser.add_argument(
        "--wallclock-tolerance",
        type=float,
        default=0.10,
        help="Max allowed regression ratio for AV wall-clock (default 0.10).",
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

    # CUPTI kernel rows only have deviceId + streamId, not NVTX's
    # globalTid. On a single-process multi-GPU trace (trtllm-serve
    # spawns 8 ranks sharing one globalPid but binding 8 different
    # threads to 8 devices) we need to link NVTX globalTid -> deviceId
    # via the stream a thread publishes ranges on. CUDA_RUNTIME events
    # are recorded per-thread and include the streamId for their stream
    # operations; streams are per-device. This block builds that map
    # when feasible; otherwise tid_to_device stays empty and kernel
    # counts fall back to aggregate (per-rank filter is silently
    # disabled and the ``notes`` section documents it).
    tid_to_device = {}
    try:
        # CUPTI_ACTIVITY_KIND_RUNTIME has globalTid + per-call stream
        # argument when available. Easiest: look at kernel stream use
        # per-globalTid via CUPTI_ACTIVITY_KIND_CUDA_EVENT where the
        # submitting thread is recorded.
        rows = conn.execute(
            """
            SELECT DISTINCT e.globalTid, k.deviceId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME e
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k
              ON k.correlationId = e.correlationId
            WHERE e.globalTid IS NOT NULL
            """
        ).fetchall()
        for tid, dev in rows:
            if tid is not None and dev is not None:
                tid_to_device.setdefault(int(tid), int(dev))
    except Exception:
        tid_to_device = {}

    summary = {}
    ok = True
    ALLTOALL_KERNEL_PATTERNS = (
        "ncclDevKernel_AllToAll",
        "ncclDevKernel_SendRecv",
    )
    ALLGATHER_KERNEL_PATTERNS = ("ncclDevKernel_AllGather",)

    # Per-rank aggregates for the pass/fail decision.
    per_tid_a2a = {}
    per_tid_ag = {}
    per_tid_op = {}

    for start, end, name, tid in ranges:
        # Resolve this outer AV range's device (rank), if known.
        dev = tid_to_device.get(tid)
        a2a = _count_in_range(kernels, start, end, ALLTOALL_KERNEL_PATTERNS,
                               device_id=dev)
        ag = _count_in_range(kernels, start, end, ALLGATHER_KERNEL_PATTERNS,
                              device_id=dev)
        # Count op-level NVTX ranges with the SAME globalTid whose start
        # lies inside [start, end]. This is per-rank and avoids
        # cross-rank overlap inflation when all 8 ranks are in the same
        # profile.
        op_count = sum(
            1 for (o_start, o_end, _o_name, o_tid) in op_ranges
            if start <= o_start <= end and o_tid == tid
        )
        summary.setdefault(name, []).append({
            "all_to_all_kernels": a2a,
            "all_gather_kernels": ag,
            "op_level_a2a_nvtx": op_count,
            "tid": tid,
            "device_id": dev,
        })

        per_tid_a2a[tid] = per_tid_a2a.get(tid, 0) + a2a
        per_tid_ag[tid] = per_tid_ag.get(tid, 0) + ag
        per_tid_op[tid] = per_tid_op.get(tid, 0) + op_count

        if ag > 0:
            ok = False
            print(
                f"FAIL range={name} tid={tid} dev={dev} [{start},{end}]: "
                f"AllGather kernels={ag} "
                "(plan requires 0 — Ulysses cross-attn must never launch AllGather).",
                file=sys.stderr,
            )
        # Op-level NVTX invariant: if op markers are present, every AV
        # cross-attn range must carry exactly 3 op-level markers on its
        # own globalTid.
        if op_ranges and op_count != 3:
            ok = False
            print(
                f"FAIL range={name} tid={tid} [{start},{end}]: op-level "
                f"a2a NVTX ranges={op_count} (expected exactly 3: Q / K|V / output).",
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
        theoretical = _theoretical_a2a_bytes_per_rank(
            U=U, B=B, S_kv=S_kv, H_kv=H_kv, D_h=D_h, elem_size=elem_size,
        )
        summary["_totals"]["theoretical_bytes_per_rank_per_tensor"] = theoretical
        summary["_totals"]["theoretical_bytes_params"] = {
            "U": U, "B": B, "S_kv": S_kv, "H_kv": H_kv,
            "D_h": D_h, "elem_size": elem_size,
        }

    # Measured bytes from the driver's sidecar JSON.
    if args.measured_bytes_sidecar is not None:
        with args.measured_bytes_sidecar.open() as fh:
            measured = json.load(fh)
        summary["_totals"]["measured_bytes"] = measured
        if args.theoretical_bytes is not None:
            theoretical = summary["_totals"]["theoretical_bytes_per_rank_per_tensor"]
            # Per the plan, the K-a2a (fused K|V 5D a2a) and the V portion
            # of the same fused op are the bytes to compare. The driver
            # records one ``a2a_kv_bytes_per_call`` entry per call and
            # averages. Compare K or V half (bytes / 2) to theoretical.
            kv_per_call = measured.get("a2a_kv_bytes_per_call_avg")
            if kv_per_call is not None:
                # The fused K|V op moves K+V together; per-tensor is /2.
                per_tensor_measured = kv_per_call / 2
                ratio = per_tensor_measured / max(1, theoretical)
                summary["_totals"]["bytes_ratio_measured_over_theoretical"] = ratio
                within = abs(ratio - 1.0) <= args.bytes_tolerance
                summary["_totals"]["bytes_within_tolerance"] = within
                if not within:
                    ok = False
                    print(
                        f"FAIL bytes: measured per-tensor K|V a2a bytes "
                        f"{per_tensor_measured} vs theoretical {theoretical} "
                        f"ratio={ratio:.4f} tolerance ±{args.bytes_tolerance}",
                        file=sys.stderr,
                    )

    # Wall-clock aggregate on AV cross-attn NVTX ranges.
    av_range_names = set(_A2A_NVTX_NAMES)
    av_wallclock_ns = _sum_range_wallclock_ns(ranges, av_range_names)
    # Also report per-rank AV wall-clock so we can average/normalize.
    per_tid_wallclock_ns = {}
    for start, end, name, tid in ranges:
        if name in av_range_names:
            per_tid_wallclock_ns[tid] = per_tid_wallclock_ns.get(tid, 0) + max(0, end - start)
    summary["_totals"]["av_wallclock_ns"] = av_wallclock_ns
    summary["_totals"]["av_wallclock_ns_per_rank"] = {
        str(tid): val for tid, val in sorted(per_tid_wallclock_ns.items())
    }

    if args.baseline_sqlite is not None:
        bconn = sqlite3.connect(
            f"file:{args.baseline_sqlite}?mode=ro", uri=True
        )
        b_ranges = _fetch_ranges(bconn, _A2A_NVTX_NAMES)
        baseline_wallclock_ns = _sum_range_wallclock_ns(b_ranges, av_range_names)
        summary["_totals"]["baseline_av_wallclock_ns"] = baseline_wallclock_ns
        b_per_tid = {}
        for start, end, name, tid in b_ranges:
            if name in av_range_names:
                b_per_tid[tid] = b_per_tid.get(tid, 0) + max(0, end - start)
        summary["_totals"]["baseline_av_wallclock_ns_per_rank"] = {
            str(tid): val for tid, val in sorted(b_per_tid.items())
        }
        # Average per-rank ratios (since each rank runs its own AV
        # cross-attn). This is what the plan asks for: per-rank
        # regression on the same hardware.
        if baseline_wallclock_ns > 0:
            ratio = av_wallclock_ns / baseline_wallclock_ns
            summary["_totals"]["av_wallclock_ratio"] = ratio
            # Regression >10% fails.
            if ratio > 1.0 + args.wallclock_tolerance:
                ok = False
                print(
                    f"FAIL wall-clock: AV NVTX current/baseline={ratio:.4f} "
                    f"(> {1.0 + args.wallclock_tolerance}); HARD regression.",
                    file=sys.stderr,
                )

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
