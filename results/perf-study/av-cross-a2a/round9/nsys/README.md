# Round 9 nsys artifacts

The `.nsys-rep` and `.sqlite` files for the perf-study and baseline
1x8 profiles are kept locally but **NOT committed** (each is ~70-270 MB,
total ~470 MB).

Reproduce by running:

```bash
# Fetch from prenyx
bash /Users/yaoyangz/projects/cc-skills/plugins/aigv-bench-dev1/scripts/bench-visual-gen/fetch-nsys.sh prenyx round9-nsys-perf_20260422_002932
bash /Users/yaoyangz/projects/cc-skills/plugins/aigv-bench-dev1/scripts/bench-visual-gen/fetch-nsys.sh prenyx round9-nsys-baseline-v2_20260422_012719

# Move into round9 tree
cp results/bench-visual-gen/round9-nsys-perf_20260422_002932/ltx2-t2v-sfp4-vanilla-1x8-cache0-tcompile1-cg0/nsys/profile.{sqlite,nsys-rep} results/perf-study/av-cross-a2a/round9/nsys/perf-study/
cp results/bench-visual-gen/round9-nsys-baseline-v2_20260422_012719/ltx2-t2v-sfp4-vanilla-1x8-cache0-tcompile1-cg0/nsys/profile.{sqlite,nsys-rep} results/perf-study/av-cross-a2a/round9/nsys/baseline/

# Run audit
python3 tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_audit.py \
  results/perf-study/av-cross-a2a/round9/nsys/perf-study/profile.sqlite \
  --baseline-sqlite results/perf-study/av-cross-a2a/round9/nsys/baseline/profile.sqlite \
  > results/perf-study/av-cross-a2a/round9/verify/audit-perf-vs-baseline.json \
  2> results/perf-study/av-cross-a2a/round9/verify/audit-perf-vs-baseline.err
```

Remote SLURM jobs for reproduction:

- task22 container build: job 2110160 → `images/perf-study-patches-r9.sqfs`
- task23 sweep: jobs 2110262/3/4/7 → `round9-a2a-sweep_20260422_002811`
- task24 perf-study nsys: job 2110268 → `round9-nsys-perf_20260422_002932`
- task24 baseline nsys: job 2110712 → `round9-nsys-baseline-v2_20260422_012719`
