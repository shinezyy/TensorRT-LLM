# Round 9 4-row a2a-tests sweep artifacts

This directory contains the fetched per-test outputs for the four rows
of `a2a-tests.csv` that Round 9's task23 executed via the
`visual-bench:bench-visual-gen` skill (batch tag
`round9-a2a-sweep_20260422_002811`).

## Why no `.avi` files

Round 9's NOTES.md stated that the 4 rows "produced `.avi` output."
That phrasing was inaccurate and is retracted in Round 10.

The LTX2-T2V sweep is run through `trtllm-serve`'s OpenAI-compatible
videos endpoint using `benchmark_visual_gen.py --num-prompts 1 ...
--save-result`. The benchmark client issues one HTTP request, receives
the generated video bytes in the response body, and records timing +
metadata into the `openai-videos-*.json` result file. It does NOT
write the video body to disk (no `--save-video` style flag exists on
this code path in the current `trtllm-new-visualgen` workspace), and
the server side streams the bytes without persisting them either.

The server logs DO confirm the videos were actually generated
(VAE decode + encoder loaded; denoising completed; status=OK in the
benchmark JSON), so the sweep did its job end-to-end — the artifact
that was never produced on-disk is the video file, not the run.

## What is in this directory

Per test (one row of the sweep):

| File                                             | Role                                              |
|--------------------------------------------------|---------------------------------------------------|
| `server.log`                                     | `trtllm-serve` server stdout/stderr               |
| `benchmark.log`                                  | `benchmark_visual_gen.py` client stdout/stderr   |
| `serve_config.yaml`                              | Synthesized serve config (attention/linear/parallelism) |
| `openai-videos-infqps-concurrency1-...-*.json`   | Benchmark JSON with `request_throughput`, `mean_e2e_latency_ms`, etc. |
| `job*.err` / `job*.out` / `job.log`              | SLURM job-level stdout/stderr                    |

Tests:

- `ltx2-t2v-sfp4-vanilla-1x2-cache0-tcompile1-cg0/` — 1 cfg × 2 ulysses = 2 GPUs
- `ltx2-t2v-sfp4-vanilla-1x8-cache0-tcompile1-cg0/` — 1 cfg × 8 ulysses = 8 GPUs
- `ltx2-t2v-sfp4-vanilla-2x2-cache0-tcompile1-cg0/` — 2 cfg × 2 ulysses = 4 GPUs
- `ltx2-t2v-sfp4-vanilla-2x4-cache0-tcompile1-cg0/` — 2 cfg × 4 ulysses = 8 GPUs

All four tests completed successfully:

| Tag | Duration (s) | Status | Source |
|-----|------|--------|--------|
| 1x2 | 22.89 | OK (Successful requests: 1, Failed: 0) | `server.log` |
| 1x8 | 20.05 | OK | benchmark JSON `duration` |
| 2x2 | 14.87 | OK | `server.log` |
| 2x4 | 14.99 | OK | `server.log` |

## Remote paths

Source on prenyx:
`/lustre/fsw/coreai_comparch_infbench/yaoyangz/aigv-results/results/bench-visual-gen/round9-a2a-sweep_20260422_002811/`

Fetch command (exactly what was run for this directory):

```bash
rsync -avz --include='*/' \
  --include='server.log' --include='benchmark.log' \
  --include='serve_config.yaml' --include='openai-videos-*.json' \
  --include='job.log' --include='job_*.err' --include='job_*.out' \
  --include='bundle_*.err' --include='bundle_*.out' \
  --exclude='*' \
  prenyx:/lustre/fsw/coreai_comparch_infbench/yaoyangz/aigv-results/results/bench-visual-gen/round9-a2a-sweep_20260422_002811/ \
  results/perf-study/av-cross-a2a/round9/avi/
```

The `avi/` directory name is kept for continuity with the Round 9
plan-section wording ("fetched `.avi` bundle under the run tag"), but
the actual fetched artifacts are the four tests' logs, serve configs,
and benchmark JSONs — not video files.
