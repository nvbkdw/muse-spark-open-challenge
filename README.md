# Muse Spark Open kernel challenge — `muse_bf16_proj`

Kernel-track entry for [openkernelchallenge.dev](https://openkernelchallenge.dev) problem `muse_bf16_proj`
(dense BF16 projections + FP32 router, one H200, timed against SGLang).

* `muse_bf16_proj/submission.py` — the submission (single file, CUDA compiled with `load_inline`).
* `muse_bf16_proj/{reference,checks,baseline}.py`, `TASK.md` — problem files written by `okc init`.
* `results/judge_runs.md` — every scored/check judge result (per shape), `results/probe*_h200.log` — on-judge
  configuration sweeps, `results/local_*` — local H100 correctness / benchmark logs.
* `tools/` — local judge-like benchmark (`bench.py`), correctness sweep (`check.py`), robustness test
  (`robust.py`), per-shape timing (`micro.py`, `timing2.py`), HBM streaming microbenchmark (`stream_bench.py`),
  and the diagnostic-probe generator used to tune on the judge GPU (`make_probe.py` + `sweep*.py`).
  `tools/experiments/` holds a parked wgmma experiment that is not part of the submission.

## Reproduce

```bash
export TORCH_EXTENSIONS_DIR=/scratch-nvme/okc/ext        # build cache on the scratch disk
python tools/check.py                                     # 189 shapes vs the FP64 reference checks
python tools/robust.py                                    # other weight seeds + adversarial weights
python tools/bench.py --reps 20                           # judge-like: CUDA graphs, cold L2, paired vs cuBLAS
okc check muse_bf16_proj/submission.py --json             # the real judge (H200)
# diagnostic sweep on the judge GPU (prints a timing table from prepare(); check runs only):
python tools/make_probe.py tools/sweep10.py /tmp/probe.py && okc check /tmp/probe.py --json
```

See the pull request description for the design, measurements and open issues.
