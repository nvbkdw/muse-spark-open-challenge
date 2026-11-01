# Muse Spark Open kernel challenge entries

Kernel-track entries for [openkernelchallenge.dev](https://openkernelchallenge.dev), developed on a local H100 and
scored on the judge's H200.

## `muse_prefill_layer` (whole decoder layer at prefill shapes)

* `muse_prefill_layer/submission.py`: the submission (single file). The pipeline is:
  * cuBLAS for the dense projections;
  * one fused Triton kernel for QK-norm, RoPE, FP8 q and the FP8 K/V cache store;
  * a custom FP8 GQA-packed wgmma attention kernel (CUDA source compiled in-process with NVRTC) for long
    full-attention layers, and FA3 (sgl_kernel) elsewhere;
  * row-chunked Triton boundary kernels (post-norm, residual gate, pre-norm);
  * Triton routing (sigmoid, biased top-8, renormalization, slab-slot assignment), dispatch, SiLU·up and combine kernels;
  * the routed experts as **humming** NVFP4 x BF16 grouped-masked wgmma GEMMs (humming is in the judge image):
    two launches per GEMM (large tile for experts with >32 routes, small tile for the rest), run concurrently.
* `muse_prefill_layer/{reference,checks,baseline,nvfp4}.py`, `TASK.md`, `AGENTS.md`: problem files from `okc init`.
* `results/prefill_judge_runs.md`: every judge run, per shape. `results/prefill_*_profile_h200.txt`: judge profiles
  (SGLang baseline and our latest submission). `results/prefill_probe_moe_h200.log`: on-judge diagnostics (real expert
  routing distributions, boundary-kernel variants, humming config sweep on the real counts).
* `tools/prefill/`:
  * `judge.py`: local judge-like harness (reference checks + CUDA-graph replay timing with a cold L2).
  * `robust.py`: extra correctness cases (kv_scale 1/64, spread lengths, page 64, odd lengths).
  * `prof.py`: per-kernel profile of one case.
  * `make_probe.py`: generates an on-judge diagnostic check run.
  * `bench_*.py`, `sweep_*.py`: microbenchmarks behind each decision.
  * `attn/`: the attention kernel's development (`attn_impl.py`, benchmarks).
  * `moe_kernel/`: a custom NVFP4 grouped-GEMM experiment; correct, but not faster than humming end to end, so not used.

```bash
export TORCH_EXTENSIONS_DIR=/scratch-nvme/okc/ext
python tools/prefill/judge.py --cases tests,scored        # 5 judge tests + 8 scored shapes, correctness + timing
python tools/prefill/robust.py                             # extra correctness cases
okc check muse_prefill_layer/submission.py --json          # the real judge (H200)
python tools/prefill/make_probe.py probes/probe_moe.py && okc check probes/probe_moe.py --json   # H200 diagnostics
```

## `muse_bf16_proj` (dense BF16 projections + FP32 router)

* `muse_bf16_proj/submission.py`: the submission (single file, CUDA compiled with `load_inline`).
* `results/judge_runs.md`, `results/probe*_h200.log`, `results/local_*`: judge results, on-judge sweeps, local logs.
* `tools/`: `bench.py`, `check.py`, `robust.py`, `micro.py`, `timing2.py`, `stream_bench.py`, `make_probe.py` + `sweep*.py`.

```bash
python tools/check.py && python tools/robust.py && python tools/bench.py --reps 20
okc check muse_bf16_proj/submission.py --json
```

See the pull request descriptions for design, measurements and open issues.
