# AC-2 Cluster Pytest Validation — Rounds 3 + 4

Prenyx 2xB200 pytest runs of `TestAVCrossAttnRealisticParity::test_av_cross_attn_parity_realistic_scale`. All jobs use the r12 container image, `coreai_comparch_inferencex` account, `srun --mpi=pmix` with `TLLM_DISABLE_MPI=1`, and `python -m pytest -v -s -p no:cacheprovider` per `BL-20260421-prenyx-pytest-stack`.

## Archived Logs

### R3 — Commit B v1 (num_layers=4, iterations=20, pressure=100 MB)

- `pre/task4-fail-bb414983c-plus-B.log` — JOBID 2116900. State: `<frozen-base>+Commit-B-v1`. pytest exit 1 via `assert_close`. Mismatch: 950/32768 (2.9%), max abs 0.03125, max rel 21.375, `any_nan=False any_inf=False` on both vx and ax.
- `post/task4-pass-bb414983c-plus-AB-ATTEMPT.log` — JOBID 2116911. State: `<frozen-base>+Commit-A+Commit-B-v1`. pytest exit 1 (expected 0). IDENTICAL mismatch pattern to the FAIL run (same 950/32768, same index, same diffs, same any_nan=False).

### R4 — Commit B v2 (num_layers=8, iterations=20, pressure=100 MB)

- `pre/task4-fail-v2-numlayers8.log` — JOBID 2119639. `<frozen-base>+Commit-B-v1+Commit-B-v2`. pytest exit 1 via `assert_close`. Mismatch: 2121/32768 (6.5%), max abs 0.0390625, max rel 51.25, `any_nan=False any_inf=False`.
- `post/task4-pass-v2-numlayers8-ATTEMPT.log` — JOBID 2119645. `<frozen-base>+Commit-A+Commit-B-v1+Commit-B-v2`. pytest exit 1. IDENTICAL mismatch pattern (same 2121, same index, same diffs).
- `pre/task-pytest-pre-v2-numlayers8.log` — JOBID 2119640. Separate-JOBID pytest-pre (AC-7 discipline). Same state as JOBID 2119639. pytest exit 1, IDENTICAL 2121 mismatch.

### R4 — Commit B v3 (num_layers=8, iterations=60, pressure=500 MB)

- `pre/task4-fail-v3-iter60-press500.log` — JOBID 2119728. `<frozen-base>+Commit-B-v1+v2+v3`. pytest exit 1, IDENTICAL 2121 mismatch as v2. Diagnostics IDENTICAL to v2.
- `post/task4-pass-v3-iter60-press500-ATTEMPT.log` — JOBID 2119704. `<frozen-base>+Commit-A+Commit-B-v1+v2+v3`. pytest exit 1, IDENTICAL 2121 mismatch.
- `pre/task-pytest-pre-v3-iter60-press500.log` — JOBID 2119729. Separate-JOBID pytest-pre at v3. IDENTICAL 2121 mismatch.

## Findings

1. **Bug's production NaN signature does not trigger at any authored test scale.** All six FAIL runs (v1 × num_layers=4, v2 × num_layers=8 × 2 separate JOBIDs, v3 × num_layers=8 + iters=60 + pressure=500 MB × 2 separate JOBIDs) produce finite output with `any_nan=False any_inf=False`, std well above 1e-4. The pure-black / NaN / inf signature only occurs at production scale (48 layers, 720×1280×121 frames, full pipeline with serve harness and KV cache).
2. **FAIL and PASS runs produce IDENTICAL outputs at every escalation step.** Same 950 elements mismatched at num_layers=4 on both branches; same 2121 elements mismatched at num_layers=8 (regardless of iterations/pressure), at the SAME indices, with the SAME absolute and relative differences. The overlap branch is numerically indistinguishable from the serial branch at these scales.
3. **Scaling iterations (20→60, 3×) and pressure tensor (100 MB→500 MB, 5×) does not change the mismatch pattern.** The per-forward U=1-vs-U=2 bf16 accumulation-order noise is NOT a function of iteration count or allocator pressure; only the final iteration's output is compared, and that output is deterministic once model + weights + inputs + Commit A state are fixed. Additional iterations give more *chances* for the race to trigger on the FAIL side, but the race is not triggering at any count.
4. **Intrinsic bf16 U=1-vs-U=2 divergence exceeds the authored 1e-2 tolerance at every layer count that's been run.** num_layers=4 gives max abs 0.031 (1.28× 1e-2 tolerance) from legitimate accumulation-order noise, num_layers=8 gives 0.039 (1.5× tolerance); PASS side cannot meet `assert_close(rtol=1e-2, atol=1e-2)` at either scale. Shape growth (the last remaining ladder step) is expected to increase this noise further, not decrease it.

## AC-2 Status

**NOT CLOSED.** Rounds 3 and 4 together confirm the test design is incompatible with the DEC-6 HARD tolerances when used against a U=1 reference at realistic multi-GPU scale:

- The bug's *actual* signature (NaN contamination → pure-black) is production-scale-only. No authored test scale (up through 8 layers + 3× iterations + 5× pressure) triggers it.
- The `assert_close(rtol=1e-2, atol=1e-2)` gate, when comparing U=2 Ulysses against U=1 reference, trips on legitimate bf16 accumulation-order noise ≈ 0.04 max abs, on both FAIL and PASS branches. It cannot discriminate bug-from-fix at any authored scale.
- Only `isfinite` + `std > 1e-4` (the two other HARD assertions) actually track the bug's production signature, but at authored scale they do not trigger on the FAIL path either.

## R5 Plan Evolution Request

Close AC-2 by revising the test strategy (Commit B v4) along one of:
- **Option P** (preferred): drop `assert_close` from the realistic-scale test; keep `isfinite` + `std > 1e-4` as the bug discriminators. Rationale: these are the bug's actual production-scale signature, and `assert_close` at 1e-2 cannot be satisfied at any realistic bf16 U=1-vs-U=2 scale. Authorize in the plan by amending DEC-6's "HARD" column: mark tolerances TREND rather than HARD for the realistic-scale test specifically; keep them HARD for the existing small-scale parity tests that DO pass at 1e-2 today.
- **Option Q**: replace the U=1 reference with a within-U=2 reference (e.g., a golden output serialized at Commit-A's known-good state). Positive: `assert_close(U=2_current, U=2_golden, rtol=1e-2, atol=1e-2)` would be tight enough because both sides have identical bf16 accumulation order. Negative: brittle to legitimate future changes that shift numerics, even when correct.
- **Option R**: rely on a cluster-bench brightness gate (`verify_video_not_dark.py`) INSTEAD of the pytest at realistic scale. task6 (R4 JOBID 2119590) already demonstrates this is sufficient to catch the bug. This moves AC-2's validation from pytest to the Option-A bench chain.

I recommend Option P: it preserves the pytest-based AC-2 gate, uses only the signals that actually track the bug, matches what the round12 production artifacts showed (mean=std=0 is the real signature), and doesn't add a new golden-file maintenance surface. A 2-line commit removes `assert_close` and leaves the other two assertions intact.

See `../bisect/v1-verdict.md` for AC-1 (task2) and `t-sweep-pre-*` for task6 / AC-3 progress.
