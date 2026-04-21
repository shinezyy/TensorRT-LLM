# U=8 SLURM drivers for AC-5 / AC-10

These in-tree drivers replace the out-of-tree `/lustre/.../ac5_u8_driver.py`
and `ac10_driver.py` referenced by Round 3 evidence logs. They must be
invoked via `srun -n N --ntasks-per-node=N --mpi=pmix` because
`torch.multiprocessing.spawn(nprocs=8)` inside a single SLURM task hits
an OMPI/PMIx abort (see `.humanize/bitlesson.md` — `BL-20260421-mp-spawn-pmix-at-u8`).

## Reproducing the U=8 AC-5 / AC-5.2 tests (prenyx)

```bash
salloc -A coreai_comparch_inferencex -p batch -N1 --exclusive -t 4:00:00 --no-shell
# Note the JOB_ID.

JOB_ID=<id>
IMG=/lustre/fsw/coreai_comparch_infbench/yaoyangz/images/70805eb64-Apr17.sqfs
WS=/lustre/fsw/coreai_comparch_infbench/yaoyangz/visualgen-ltx2

for T in ac5_full ac52_audio_attn1 ac5_neg_eager_strip ac5_neg_missing_a2v_mask; do
  srun --jobid=$JOB_ID --mpi=pmix -n 8 --ntasks-per-node=8 \
       --container-image=$IMG --container-name=r4 \
       --container-mounts=/lustre/:/lustre/ \
       bash -c "cd $WS && source .venv-3.12/bin/activate && \
       TLLM_DISABLE_MPI=1 python \
       tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac5_u8_driver.py $T"
done
```

## Reproducing the AC-10 divergent-flags test

```bash
srun --jobid=$JOB_ID --mpi=pmix -n 2 --ntasks-per-node=2 \
     --container-image=$IMG --container-name=r4 \
     --container-mounts=/lustre/:/lustre/ \
     bash -c "cd $WS && source .venv-3.12/bin/activate && \
     TLLM_DISABLE_MPI=1 python \
     tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac10_driver.py"
```

## Why these are driver scripts, not `pytest` entry points

`pytest`'s collect → fixture → call lifecycle does not play well with
`mp.spawn(nprocs=8)` on prenyx — either the spawn-based U=8 test hits
the PMIx abort, or `python -m pytest` swallows per-rank stdout under
`pytest-shard`/`pytest-xdist`. Running these drivers as plain Python
under `srun -n N` gives each rank its own PMIx context and preserves
per-rank stdout.
