# Round 9 AV cross-attn outlier analysis

## Context

Round 9's first audit reported 2 out of 4652 `ltx2.v2a_cross_attn`
NVTX ranges with `op_level_a2a_nvtx=0` (zero inner Q / K|V / output
op-level markers). The Round 9 summary and NOTES.md labelled these
"sub-100us startup artifacts." Codex's Round 9 review correctly
rejected that framing: the capture window was steady-state steps 5-10
(`PROFILE_START_STEP=5`, `PROFILE_NUM_STEPS=6`), and the two outliers
are at the TAIL of the trace, not the head.

This note documents what the two outliers actually are.

## Direct sqlite probe (Round 10)

```
sqlite: results/perf-study/av-cross-a2a/round9/nsys/perf-study/profile.sqlite
```

The max NVTX `end` timestamp across the entire trace is
**470 200 777 092 563 ns**. Eleven NVTX ranges end at this exact
nanosecond:

```sql
SELECT COUNT(*) FROM NVTX_EVENTS WHERE end = 470200777092563;
-- 11
```

Both `v2a` outliers are in that set:

| range                  | tid (rank) | device | start              | end                | dur      | op_count |
|------------------------|------------|--------|--------------------|--------------------|---------:|---------:|
| `ltx2.v2a_cross_attn` | 323437413673508 (rank 6) | 6 | 470 200 775 613 887 | 470 200 777 092 563 | 1478.7us |        0 |
| `ltx2.v2a_cross_attn` | 323437346564640 (rank 2) | 2 | 470 200 777 039 187 | 470 200 777 092 563 |   53.4us |        0 |

## Mechanism

`bench-nsys-job.sh` runs the workload under `nsys profile -c
cudaProfilerApi --capture-range-end=stop` and the workload itself calls
`torch.cuda.cudart().cudaProfilerStop()` at the end of the configured
profile window. On `cudaProfilerStop()` nsys terminates every NVTX
range still open on every thread at the exact stop timestamp. The
Python code inside `UlyssesCrossAttention.forward` opens the three
op-level NVTX ranges AFTER entering `ltx2.v2a_cross_attn`, so if the
profiler stop lands between the outer `range_push("ltx2.v2a_cross_attn")`
and the first inner `range_push("ulysses.cross.a2a.q")` on a given
rank, that rank's outer v2a range is truncated with zero inner markers.

The two outliers are exactly this: rank 6's v2a range was ~1.5ms into
its forward (a normal v2a duration) when the profile stopped; rank 2's
was ~53us in. Neither rank reached its first op-level `range_push`
call before the stop tick.

Rank 2's AV cross-attn is therefore NOT a 53us micro-range — it is a
range that ran for at most 53us before being truncated by the
profiler-stop boundary. No sub-100us warmup-artifact claim is needed.

## Implications for the audit

The Round 10 audit (`ac9_audit.py`) classifies any NVTX range whose
`end` timestamp matches the shared profile cutoff as
`truncated_at_profile_cutoff=true` and excludes it from the
"exactly 3 op-level markers per AV range" assertion. This is not a
loosening of the plan's HARD invariant — the invariant is a property
of the RUNNING code, and a range that was cut off mid-flight by
`cudaProfilerStop` has no opportunity to satisfy or violate it. The
4650 non-truncated ranges all carry exactly 3 op-level markers on
their owning rank.

## Implications for the round

The Round 9 summary's "sub-100us startup artifacts" framing is
incorrect and has been corrected in the Round 10 summary + NOTES.md
update. The underlying evidence — that 4650 of 4652 ranges carry the
exact 3-marker invariant — is unchanged, but the 2 remaining ranges
are re-classified as profiler-stop truncated rather than mystery
outliers.
