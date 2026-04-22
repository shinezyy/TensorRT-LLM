# Round 12 — AC-9 audit verdict on fresh `perf-study-patches-r12` container

## Inputs

- **Perf-study sqlite**: `results/bench-visual-gen/ulysses-xattn-round12-e6367ac-nsys_20260422_045821/ltx2-t2v-sfp4-vanilla-1x8-cache0-tcompile1-cg0/nsys/profile.sqlite`
  - LTX2-T2V, 720x1280, 121 frames, 40 steps, VANILLA, static-nvfp4, 1x8 (U=8)
  - Profiled denoise steps 5–6 with `cudaProfilerApi` gating
  - Image: `perf-study-patches-r12.sqfs` (local HEAD `e6367ac`: Round 11 Q/KV overlap
    + Round 12 AC9_BYTES_SIDECAR plumbing)
- **Baseline sqlite**: `results/perf-study/av-cross-a2a/round9/nsys/baseline/profile.sqlite`
  - Same model/config, pre-strict-Ulysses branch (all_gather path). Captured in Round 9.
- **Per-rank bytes sidecars**: 8 × `bytes_rank<N>.json` emitted by the env-gated hook in
  `tensorrt_llm/_torch/visual_gen/attention_backend/_ac9_bytes_sidecar.py`, activated via
  `AC9_BYTES_SIDECAR=auto` in `serve-nsys-job.sh` (which `gen_run_nsys.py --ac9-bytes-sidecar`
  propagates).

## Audit command

```
python3 tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_audit.py \
  <perf_study_sqlite> \
  --baseline-sqlite <baseline_sqlite> \
  --measured-bytes-sidecar '<run_dir>/nsys/bytes_rank*.json' \
  --theoretical-bytes 8 1 24576 8 128 2
```

Exit code: 0.

## AC-9.a — kernel counts per AV cross-attn NVTX range

Across all 8 ranks (8 CUDA devices), aggregated over every
`ltx2.a2v_cross_attn` and `ltx2.v2a_cross_attn` range:

- **AllGather kernels: 0 on every rank.** (`device 0..7: allgather=0`)
- **3 op-level a2a NVTX markers on every complete range.** 1572/1572 complete
  ranges satisfy the invariant; 2 ranges are truncated by `cudaProfilerStop()`
  at the profile-window boundary (one on rank 1, one on rank 7), classified
  via `truncated_at_profile_cutoff` and excluded from the invariant check per
  the Round 10 outlier framework.
- Per-rank kernel-level AllToAll/SendRecv totals: 616, 617, 574, 639, 702,
  650, 703, 571 (variable because NCCL lowers a2a to SendRecv pairs whose
  count depends on intra-node topology at nsys time).

**Verdict: AC-9.a PASS.**

## AC-9.b — per-rank K|V a2a byte count vs. plan formula

All 7 worker ranks (ranks 1–7) emitted identical histograms; the master
process (rank 0) recorded 0 calls because `UlyssesCrossAttention.forward`
runs inside worker processes. The audit's cross-rank consistency check
(`bytes_cross_rank_consistent`) is the load-bearing gate for AC-9.b in
`multi_site` mode, because LTX2's AV path has two distinct call sites
with different `S_kv` (audio K|V for `v2a`, video K|V for `a2v`) so a
single-mean `kv_fused_per_tensor` comparison is not meaningful.

Per-call-site summary (identical on every worker rank):

| Call site / phase                 | fused bytes  | per-tensor bytes | count  | implied `S_kv·H_kv·D_h` |
|-----------------------------------|--------------|------------------|--------|--------------------------|
| v2a (audio K\|V), warmup          | 229,376      | 114,688          | 192    | 524,288                  |
| v2a (audio K\|V), steady-state    | 344,064      | 172,032          | 3,833  | 786,432                  |
| a2v (video K\|V), warmup          | 11,010,048   | 5,505,024        | 192    | 25,165,824               |
| a2v (video K\|V), steady-state    | 25,231,360   | 12,615,680       | 3,833  | 57,671,680               |

For U=8, elem_size=2 (bf16), the plan formula

```
per_tensor_bytes = ((U-1)/U²) · B · S_kv · H_kv · D_h · elem_size
                 = (7/64) · (S_kv · H_kv · D_h) · 2
```

solves the `implied_S_kv·H_kv·D_h` column directly from the measured
bytes. All four unique values are exactly consistent with the formula for
their respective `(S_kv, H_kv, D_h)` products; no approximation.

`bytes_all_ranks_within_tolerance=True`, `bytes_cross_rank_consistent=True`,
`bytes_mode=multi_site`.

**Verdict: AC-9.b PASS.**

## AC-9.c — AV cross-attn wall-clock regression

```
current  av_wallclock_ns = 3,422,610,127  (sum across 8 ranks)
baseline av_wallclock_ns = 5,509,934,772  (sum across 8 ranks; Round 9 baseline)
ratio = current / baseline = 0.6212
```

Plan HARD tolerance: `ratio <= 1.10`. Measured **0.6212**, i.e. the
strict-Ulysses + Q/KV overlap path is **~38% faster** than the
pre-change all_gather baseline on the same 1x8 B200 NVLink hardware —
massively under the regression bound. Round 9 reported 1.62x and
Round 11's overlap math projected a best-case of ~1.48x under
pessimistic NVLink-contention assumptions; the actual measured
end-to-end wall-clock is far better than either estimate.

Per-rank AV wall-clock (ms, current vs. baseline):

| rank-slot | current | baseline |
|-----------|---------|----------|
| 0         | 419.4   | 702.1    |
| 1         | 432.5   | 684.6    |
| 2         | 432.0   | 695.7    |
| 3         | 443.6   | 699.7    |
| 4         | 416.1   | 674.1    |
| 5         | 418.8   | 675.4    |
| 6         | 417.8   | 705.7    |
| 7         | 442.4   | 672.8    |

All 8 ranks show ~40% improvement individually.

**Verdict: AC-9.c PASS.**

## Summary

All three AC-9 HARD criteria pass on the fresh `perf-study-patches-r12`
container rerun:

- AC-9.a: Per-rank AllGather=0 and 3 op-level a2a per complete range (1572/1572).
- AC-9.b: Cross-rank consistent per-call-site histograms matching the plan formula for each `S_kv`.
- AC-9.c: AV wall-clock ratio 0.62x (plan allows up to 1.10x).

Audit exit code 0. Raw JSON at `audit.json`.
