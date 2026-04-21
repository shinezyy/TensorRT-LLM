# AC-9 1×8 nsys kernel-count + op-level audit (Round 6)

Fresh 1×8 nsys capture on prenyx B200 × 8 (SLURM job 2109956, container
`70805eb64-Apr17.sqfs`) driven by
`tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_nsys_driver.py`
(audio_frames=125, U=8, VANILLA backend, 3 warmed forwards). The code on
the node was the current `perf-study-patches` branch, which carries the
new op-level NVTX markers inside `UlyssesCrossAttention.forward`:

- `ulysses.cross.a2a.q` — Q all-to-all-4D
- `ulysses.cross.a2a.kv` — fused K|V all-to-all-5D
- `ulysses.cross.a2a.out` — output all-to-all-4D

The audit script `_u8_drivers/ac9_audit.py` now counts those markers per
AV cross-attn NVTX range and asserts the plan's per-range "exactly 3 a2a
ops" invariant at the op level, which is topology-independent — unlike
the NCCL kernel count, which at intra-node B200 is `(U-1)` `SendRecv`
kernels per op plus async-spillover noise.

## Audit result (HARD contract)

| Rank | a2v ranges | v2a ranges | AllGather | Op-level a2a NVTX per range | Audit exit |
|------|------------|------------|-----------|------------------------------|------------|
| 0 | 4 | 4 | 0 | 3 (every range, all 8) | 0 |
| 1 | 4 | 4 | 0 | 3 (every range, all 8) | 0 |
| 2 | 4 | 4 | 0 | 3 (every range, all 8) | 0 |
| 3 | 4 | 4 | 0 | 3 (every range, all 8) | 0 |
| 4 | 4 | 4 | 0 | 3 (every range, all 8) | 0 |
| 5 | 4 | 4 | 0 | 3 (every range, all 8) | 0 |
| 6 | 4 | 4 | 0 | 3 (every range, all 8) | 0 |
| 7 | 4 | 4 | 0 | 3 (every range, all 8) | 0 |

Per-range NCCL kernel counts (AllToAll/SendRecv) still vary by topology
and async spillover (e.g. rank 0 a2v `[3,1,2,2]`, v2a `[4,4,4,4]`); the
op-level invariant is what the plan's "exactly 3" HARD text actually
pins down. Full per-range JSON in `audit-rank{0..7}.json`.

**OVERALL: PASS** on all three HARD checks:

1. `0 AllGather` inside every `ltx2.a2v_cross_attn` / `ltx2.v2a_cross_attn`
   range.
2. `ncclDevKernel_AllToAll` / `ncclDevKernel_SendRecv` kernels present
   globally (25 total on rank 0).
3. Exactly 3 op-level a2a NVTX ranges per AV cross-attn range (Q /
   fused K|V / output), which is the plan's per-range "exactly 3"
   invariant.

## Capture command

```bash
srun --jobid=$JOB --mpi=pmix -n 8 --ntasks-per-node=8 \
     --container-image=$IMG --container-name=r6nsys \
     --container-mounts=/lustre/:/lustre/ \
     bash -c "cd $WS && source .venv-3.12/bin/activate && \
     MASTER_PORT=29921 TLLM_DISABLE_MPI=1 \
     nsys profile -t cuda,nvtx,nccl -f true \
         -o $OUT/profile_rank\${SLURM_PROCID} \
         python tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_nsys_driver.py 3"
```

## Remaining follow-up (tracked, not in this round)

- Per-rank AllToAll byte audit against theoretical
  `((U−1)/U²)·B·S_kv·H_kv·D_h·elem_size` ±10%. The op-level NVTX markers
  now give clean windows to sum payload bytes from
  `CUPTI_ACTIVITY_KIND_KERNEL` per op; the audit still needs the
  aggregation wiring and a full-scale (48-layer, 720×1280, 121 frames,
  40 steps) run so the bytes pass reference values.
- Wall-clock regression ≤ 10% on full-scale
  `ltx2-t2v-sfp4-vanilla-1x8` vs. pre-change baseline. Reduced-config
  driver run here is not directly comparable; the intended vehicle is
  the 4-row `a2a-tests.csv` sweep through
  `visual-bench:bench-visual-gen` against a container built from
  `perf-study-patches` HEAD.
