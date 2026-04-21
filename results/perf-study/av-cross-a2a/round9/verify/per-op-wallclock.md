# AC-9 per-op wall-clock breakdown (Round 9 sqlite)

Per Codex's Round 10 review step 1: use the `ulysses.cross.a2a.{q,kv,out}`
NVTX markers already in the Round 9 perf-study trace to determine the
dominant Ulysses-introduced phase of the 1.62x AV wall-clock regression.

## Source

- `results/perf-study/av-cross-a2a/round9/nsys/perf-study/profile.sqlite`
  (1x8 LTX2-T2V, 720x1280, 40 steps, Ulysses=8, perf-study branch).
- Rank attribution via `globalPid = globalTid & ~((1<<24)-1)` +
  `CUPTI_ACTIVITY_KIND_KERNEL.globalPid -> deviceId` (verified 1:1 in
  Round 10).

## Aggregate per-op stats across all 4651 non-truncated calls

| Op                       | mean    | p50     | p99     | total (8 ranks) |
|--------------------------|---------|---------|---------|-----------------|
| `ulysses.cross.a2a.q`   | 216.4us | 204.8us | 695.4us | 1.01 s |
| `ulysses.cross.a2a.kv`  | 167.2us | 163.0us | 253.9us | 0.78 s |
| `ulysses.cross.a2a.out` | 177.7us | 171.8us | 230.4us | 0.83 s |
| **SUM**                 | **561.3us** | 540us | 1179us | **2.62 s** |

All three a2a ops contribute within a ~1.3x band of each other (Q is
the largest at 216us, K|V the smallest at 167us).

## Per-rank per-op totals (ms, all 6 profiled denoise steps)

| rank | Q a2a   | K\|V a2a | OUT a2a | SUM    |
|-----:|--------:|---------:|--------:|-------:|
| 0    | 127.9ms | 95.2ms   | 102.8ms | 325.9ms |
| 1    | 132.9ms | 100.8ms  | 108.2ms | 341.9ms |
| 2    | 126.2ms | 99.2ms   | 105.1ms | 330.4ms |
| 3    | 129.9ms | 98.8ms   | 101.4ms | 330.1ms |
| 4    | 128.7ms | 96.8ms   | 105.3ms | 330.8ms |
| 5    | 118.8ms | 95.5ms   | 99.7ms  | 314.0ms |
| 6    | 122.2ms | 95.3ms   | 98.9ms  | 316.4ms |
| 7    | 120.0ms | 96.2ms   | 105.3ms | 321.4ms |
| **avg** | **125.8ms** | **97.2ms** | **103.3ms** | **326.4ms** |

## How much of the regression is the 3 a2a ops?

- Per-rank AV NVTX wall-clock (perf-study, sum of `a2v + v2a`): 1116 ms
  (per-rank mean across 8 ranks, from Round 10 audit output).
- Per-rank AV NVTX wall-clock (baseline, AllGather-based): 689 ms.
- Per-rank AV regression: `1116 - 689 = 427 ms`.
- Per-rank SUM of 3 a2a ops: `325.9 + 100.8 + ... / 8 = 326 ms`.
- 326 / 427 = **76%** of the regression is directly inside the three
  a2a collectives; the remaining ~24% is increased attention-compute
  and NCCL-kernel scheduling overhead under the new pattern.

Takeaway: removing or overlapping ANY of the 3 a2a ops is wall-clock
relevant. No single op dominates. The largest saving is obtainable by
overlapping Q a2a (216us) with K|V a2a (167us): those two ops are
independent (Q comes from `audio_hidden_state`, K|V comes from
`video_hidden_state` in v2a, or vice versa in a2v), and on separate
CUDA streams they could execute concurrently rather than serially on
the default stream.

## Overlap potential

The current `UlyssesCrossAttention.forward` issues Q a2a first, then
K|V a2a, then attention, then OUT a2a, all on the same (default) CUDA
stream, so NCCL kernels serialize.

If Q a2a and K|V a2a are issued on different streams:

- Expected per-range saving: `max(Q_t, KV_t) - min(Q_t, KV_t) = 216us - 167us = 49us`
  at best (if both complete in parallel), OR up to `min(Q_t, KV_t) = 167us`
  of concurrency if NVLink has spare bandwidth.
- Per-rank saving: `167us x 580 ranges ≈ 97ms` (if fully overlapped) down
  to `49us x 580 ≈ 28ms` (if purely serial).
- Best case: new per-rank AV wall-clock = `1116 - 97 = 1019 ms`.
  Ratio vs baseline (689 ms) = **1.48x**. Still above the ≤1.10
  HARD tolerance but a meaningful move in the right direction.

OUT a2a is harder to overlap: its input is the attention output,
which is a data-dependent blocker. OUT a2a could in principle be
overlapped with the NEXT layer's QKV projection on the same rank, but
that requires a larger cross-module change.

## Implementation plan for Round 11

Start with the smallest shape-preserving change: Q a2a + K|V a2a on
two streams with a fence before attention. This is a within-module
edit to `UlyssesCrossAttention.forward`, does NOT change any tensor
shape or collective semantics, and is verifiable via existing AC-1/2/3
tests plus an nsys rerun.

If the rerun shows meaningful (≥20%) reduction in the per-rank AV
wall-clock regression, a follow-up round can attempt the harder OUT
a2a overlap with the next-block QKV projection.
