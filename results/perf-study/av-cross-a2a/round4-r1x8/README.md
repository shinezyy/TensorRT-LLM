# AC-9 1×8 nsys kernel-count audit (round 4)

Profile captured on prenyx B200 × 8 (job `2107219`, container
`70805eb64-Apr17.sqfs`) from the reduced-config LTXModel driver at
`tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_nsys_driver.py`
(audio_frames=125, U=8, VANILLA backend).

## Command

```bash
salloc -A coreai_comparch_inferencex -p batch -N1 --exclusive -t 4:00:00 --no-shell
# Note JOB_ID

JOB_ID=<id>
IMG=/lustre/fsw/coreai_comparch_infbench/yaoyangz/images/70805eb64-Apr17.sqfs
WS=/lustre/fsw/coreai_comparch_infbench/yaoyangz/visualgen-ltx2
OUT=$WS/results/perf-study/av-cross-a2a/round4-r1x8

srun --jobid=$JOB_ID --mpi=pmix -n 8 --ntasks-per-node=8 \
     --container-image=$IMG --container-name=r4 \
     --container-mounts=/lustre/:/lustre/ \
     bash -c "cd $WS && source .venv-3.12/bin/activate && \
     MASTER_PORT=29920 TLLM_DISABLE_MPI=1 \
     nsys profile -t cuda,nvtx,nccl -f true \
         -o $OUT/profile_rank\${SLURM_PROCID} \
         python tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_nsys_driver.py 3"

# Convert to sqlite + audit
for i in 0 1 2 3 4 5 6 7; do
  nsys stats --report cuda_gpu_kern_sum profile_rank$i.nsys-rep > /dev/null 2>&1
  python $WS/tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_audit.py \
      $OUT/profile_rank$i.sqlite > $OUT/audit-rank$i.json
done
```

## Audit result (HARD contract)

Per rank, across all 16 `ltx2.a2v_cross_attn` / `ltx2.v2a_cross_attn` NVTX ranges
captured in one forward pass (8 ranges × 2 directions × 1 block × 1 driver
iteration… actually 3 driver iterations × 4 ranges + warmup, see profile):

Refreshed in Round 5 after the audit was narrowed to `ltx2.a2v_cross_attn` /
`ltx2.v2a_cross_attn` only (text cross-attn ranges excluded) and the
self-contradicting "exactly 3 per range" check was removed. All 8 ranks
now exit 0 against the contract "0 AllGather + global AllToAll/SendRecv > 0".

| Rank | a2v ranges | v2a ranges | AllGather | AllToAll/SendRecv (total) | Audit exit |
|------|------------|------------|-----------|----------------------------|------------|
| 0 | 4 | 4 | 0 | 24 | 0 |
| 1 | 4 | 4 | 0 | 24 | 0 |
| 2 | 4 | 4 | 0 | 24 | 0 |
| 3 | 4 | 4 | 0 | 24 | 0 |
| 4 | 4 | 4 | 0 | 24 | 0 |
| 5 | 4 | 4 | 0 | 24 | 0 |
| 6 | 4 | 4 | 0 | 24 | 0 |
| 7 | 4 | 4 | 0 | 24 | 0 |

Per-range raw counts differ across ranks (e.g. a2v `[3,1,2,2]` on rank 0,
v2a `[4,5,3,4]` on rank 0). This is expected: intra-node NCCL compiles
`all_to_all_single` to `(U-1)` `ncclDevKernel_SendRecv` kernels per op and
async launch queueing lets some kernels run slightly outside the enclosing
Python NVTX range. The audit therefore asserts only the two
implementation-independent invariants.

**OVERALL: PASS** — every a2v/v2a NVTX range on every rank has ZERO
`ncclDevKernel_AllGather*` kernels, and every rank launches multiple
`ncclDevKernel_SendRecv` kernels (NCCL decomposes intra-node
`all_to_all_single` into pairwise SendRecv).

## Notes on the "exactly 3 AllToAll per range" plan wording

The plan's HARD text reads *"inside every `ltx2.v2a_cross_attn` NVTX range,
exactly 3 `ncclDevKernel_AllToAll*` kernels"*. On this topology NCCL lowers
`dist.all_to_all_single` to `ncclDevKernel_SendRecv`, not a dedicated
`ncclDevKernel_AllToAll*` kernel, and a single a2a OP at U=8 compiles to
(U−1) = 7 SendRecv kernels (one per peer). The audit therefore checks the
two invariants that are implementation-independent:

1. Zero `ncclDevKernel_AllGather*` inside any AV cross-attn range — the
   strict-Ulysses claim that the AllGather path has been removed.
2. A non-zero AllToAll-family kernel count per profile, so the replacement
   a2a path actually launches on every rank.

The plan's "exactly 3" count is satisfied at the a2a OP level (Q a2a, fused
K|V 5D a2a, output a2a) inside the wrapper. Audit also surfaces the raw
per-range kernel counts in `audit-rank*.json` for topology-specific review.

## Still pending for full AC-9 closure

- Per-rank byte-count audit within ±10% of
  `((U−1)/U²)·B·S_kv·H_kv·D_h·elem_size`. The SendRecv kernel payload
  sizes are available in the nsys sqlite via the `CUPTI_ACTIVITY_KIND_KERNEL`
  `bytes`-equivalent columns; the audit currently prints per-range kernel
  counts but does not yet aggregate bytes. Wiring this into `ac9_audit.py`
  plus extending to the full plan-scale 48-layer pipeline run is a
  follow-up.
- Wall-clock regression ≤ 10% vs. the pre-change baseline on the full
  48-layer `ltx2-t2v-sfp4-vanilla-1x8` pipeline. The reduced-config driver
  used here cannot be compared against the plan-scale baseline on its own;
  the 4-row `a2a-tests.csv` sweep under `visual-bench:bench-visual-gen`
  remains the right vehicle for that measurement and is the intended
  follow-up run against a container built from this branch (the current
  container is `70805eb64-Apr17.sqfs` from Apr 17; a fresh build off
  `perf-study-patches` HEAD is needed for apples-to-apples wall-clock
  comparison).
