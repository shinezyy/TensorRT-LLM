# AC-2 Cluster Pytest Validation — Round 3

Two prenyx 2xB200 pytest runs of `TestAVCrossAttnRealisticParity::test_av_cross_attn_parity_realistic_scale`, one against `<frozen-base> + Commit-B test patch` (expected FAIL) and one against `<frozen-base> + Commit-A + Commit-B test patch` (expected PASS).

## Cluster Execution

- **Account**: `coreai_comparch_inferencex`
- **Image**: `/lustre/fsw/coreai_comparch_infbench/yaoyangz/images/perf-study-patches-r12.sqfs`
- **Pytest invocation** (per BL-20260421-prenyx-pytest-stack): `srun --mpi=pmix ... TLLM_DISABLE_MPI=1 python -m pytest -v -s -p no:cacheprovider ...`

## Results

### pre/task4-fail-bb414983c-plus-B.log — JOBID 2116900

**State**: `<frozen-base> + Commit-B test patch` (overlap path ACTIVE; no Commit A).

**Pytest exit**: 1 (FAILED, as required for AC-2 negative).

Failure assertion: `torch.testing.assert_close(vx_uly, vx_ref, rtol=1e-2, atol=1e-2)` mismatch — 950 / 32768 elements (2.9%), greatest abs diff 0.03125 at (0, 1987, 2), greatest rel diff 21.375 at (0, 845, 1).

Diagnostics (from `_dump_all_parity_diagnostics`):
```
vx_uly: mean=-0.0708209 std=0.835307 abs_max=1.97656 any_nan=False any_inf=False
vx_ref: mean=-0.0706783 std=0.83519  abs_max=1.96875 any_nan=False any_inf=False
ax_uly: mean= 0.380604  std=1.23752  abs_max=2.65625 any_nan=False any_inf=False
ax_ref: mean= 0.380753  std=1.23762  abs_max=2.65625 any_nan=False any_inf=False
```

Neither `any_nan` nor `any_inf` triggered. Output std is far above 1e-4. **The bug's production signature (NaN/inf contamination → pure-black, mean=std=0) did NOT manifest at this test scale.**

### post/task4-pass-bb414983c-plus-AB-ATTEMPT.log — JOBID 2116911

**State**: `<frozen-base> + Commit-A + Commit-B test patch` (overlap branch removed via Commit A).

**Pytest exit**: 1 (unexpected FAIL — expected 0 PASS for AC-2 positive).

Failure assertion: identical to the FAIL run — same 950 / 32768 mismatch, same (0, 1987, 2) index, same 0.03125 / 21.375 diffs.

## Interpretation

The `_REALISTIC_AV_CONFIG` in the test uses `num_layers=4`, video 8×16×16, audio 96, text 64, bf16, 20 iterations + 100 MB pressure tensor. This scale does NOT trigger the overlap-path's NaN-propagating race at the frozen base — both the overlap branch and the serial branch produce finite output that differs from the U=1 reference by the SAME amount (identical mismatch pattern on both runs).

The 2.9% mismatch is therefore the legitimate bf16 numerical divergence between U=1 and U=2 head distribution (order-of-operations in Q·K^T accumulation), not a bug-discriminating signal.

Consequences:
- AC-2 negative proof: technically satisfied — pytest at `<frozen-base> + Commit-B` exits non-zero as required. However, the FAIL mechanism (assert_close divergence) is not the plan's intended bug catch (`isfinite` / `std > 1e-4` failure).
- AC-2 positive proof: **NOT satisfied** — pytest at `<frozen-base> + Commit-A + Commit-B` also exits 1, because the same bf16 noise trips the 1e-2 tolerance.
- Test design gap: `num_layers=4` with 8×16×16 video + 96 audio is insufficient to trigger the race-driven NaN signature that the bug produces at production scale (48 layers + 720×1280×121).

## Round 4 Remediation (required before AC-2 can close)

Either:
1. Scale the test up until isfinite / std > 1e-4 actually discriminate the bug path (e.g., increase `num_layers` from 4 toward production 48; increase iteration count; increase 100 MB pressure tensor; or grow the input shape).
2. Re-examine whether the plan's `rtol=1e-2 atol=1e-2` HARD thresholds (DEC-6) are the right discriminator at the authored scale. If the bug at production scale produces NaN and zero output, AC-2's regression gate can rely on `isfinite` + `std > 1e-4` alone; the `assert_close` should either be widened to accept bf16 U=1-vs-U=2 noise or removed.

Either path requires a new Commit B revision and another task4 cluster FAIL/PASS proof. This is tagged as a blocking side issue in the goal tracker and queued for Round 4.

## AC-2 Closure Status

**Partially met**: negative run exits 1 (required), positive run exits 1 (required 0). AC-2 is **NOT CLOSED** this round.

AC-1 (task2) IS closed independently; see `bisect/v1-verdict.md`.
