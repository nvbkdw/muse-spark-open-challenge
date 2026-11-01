"""Correctness + timing of moe_fp4 vs humming (split launches, as in submission.py).

usage: CUDA_VISIBLE_DEVICES=1 python tools/prefill/moe_kernel/test_moe_fp4.py [--no-humming] [--only T,T] [--reps 20]
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(ROOT, "muse_prefill_layer"))
sys.path.insert(0, HERE)
os.environ.setdefault("HUMMING_CACHE_DIR", "/tmp/humming_cache_okc")
os.environ.setdefault("HUMMING_TMP_DIR", "/tmp/humming_tmp_okc")

import moe_fp4  # noqa: E402
from nvfp4 import dequantize, random_weights  # noqa: E402

E = 32
dev = "cuda"

DISTS = {
    753: [528, 221, 152, 85, 68, 52, 36, 10, 3, 3, 2, 1, 1, 1, 1] + [0] * 17,
    3012: [2001, 829, 474, 352, 316, 212, 147, 30, 12, 9, 4, 3, 2, 2, 2, 1] + [0] * 16,
    16384: [11956, 4558, 1896, 1807, 1718, 922, 461, 146, 15, 11, 9, 5, 3, 1] + [0] * 18,
}


def make_counts(T, kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "real":
        c = torch.tensor(DISTS[T], dtype=torch.int32)
        return c[torch.randperm(E, generator=g)]
    if kind == "dirichlet":  # 8 routes per token spread over 256 experts, 32 local
        p = torch.distributions.Dirichlet(torch.full((256,), 0.3)).sample()
        ids = torch.multinomial(p, T * 8, replacement=True, generator=g).view(T, 8)
        c = torch.zeros(E, dtype=torch.int32)
        for e in range(E):
            c[e] = int((ids == e).any(dim=1).sum())
        return c
    if kind == "edge":
        base = [0, 1, 7, 8, 9, 15, 16, 17, 255, 256, 257, T, T - 1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                176, 177, 511, 512, 513, 1000, 0, 5]
        return torch.tensor([min(max(b, 0), T) for b in base], dtype=torch.int32)
    if kind == "uniform":
        return torch.randint(0, T + 1, (E,), generator=g, dtype=torch.int32)
    raise ValueError(kind)


def timeit(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    ts = []
    for _ in range(reps):
        flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


# ------------------------------------------------------------------ humming (copied from submission.py)
def humming_prepare(w, sf, gs):
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema
    from humming.schema.humming import HummingInputSchema
    from humming.transform import prepare_layer_config, transform_humming_tensors
    from humming import dtypes
    e, n, k2 = w.shape
    k = 2 * k2
    schema = ModeloptNvfp4WeightSchema()
    wschema, tensors = schema.convert_humming(
        tensors={"weight": w.contiguous(), "weight_scale": sf.contiguous(), "weight_scale_2": gs.contiguous()},
        shape_n_stacks=[n], shape_k_stacks=[k], param_dtype=torch.bfloat16, num_experts=e)
    ischema = HummingInputSchema(a_dtype=dtypes.bfloat16)
    cfg = prepare_layer_config(shape_n=n, shape_k=k, weight_schema=wschema, input_schema=ischema, num_experts=e,
                               pad_n_to_multiple=1, pad_k_to_multiple=1, torch_dtype=torch.bfloat16, has_bias=False)
    tensors = transform_humming_tensors(cfg, tensors)
    return {"config": cfg.to_str(), "tensors": tensors, "n": n, "k": k,
            "locks": torch.zeros(1024, dtype=torch.int32, device=dev)}


_CC = json.dumps({"gemm_type": "grouped_masked"})


def _gemm_cfg(bm, bn=256, bk=64, stages=4, stream_k=True):
    return {"block_shape": (bm, bn, bk), "warp_shape": (bm, 32, 64), "use_stream_k": stream_k,
            "use_f16_accum": False, "num_stages": stages, "use_warp_spec": True, "use_tma": True,
            "use_mbarrier": True}


def humming_run(bank, x, out, counts, tuning, valid_m):
    from humming import ops
    t = bank["tensors"]
    return ops.humming_gemm(
        layer_config=bank["config"], compute_config=_CC, tuning_config=json.dumps(tuning),
        inputs=x, weight=t["weight"], weight_scale=t["weight_scale"], weight_scale_2=t.get("weight_scale_2"),
        outputs=out, expert_layout=counts, locks=bank["locks"], top_k=1, valid_shape_m=valid_m)


def humming_split(bank, x, out, counts, T):
    big = torch.where(counts > 64, counts, 0).int()
    small = torch.where(counts > 64, 0, counts).int()
    vm = max(2 * T, 1)

    def fn():
        humming_run(bank, x, out, big, _gemm_cfg(176, 256, 64), vm)
        humming_run(bank, x, out, small, _gemm_cfg(32, 256, 128), vm)
    return fn


# ------------------------------------------------------------------ checks
def relerr_all(out, x, counts, slab, wd_fn, fused=False):
    """max and aggregate relative L2 error over all valid rows of every expert."""
    num = den = 0.0
    worst = 0.0
    for e in range(E):
        c = int(counts[e])
        if c == 0:
            continue
        wd = wd_fn(e)
        ref = x[e * slab:e * slab + c].float() @ wd.T
        if fused:
            H = ref.shape[1] // 2
            ref = torch.nn.functional.silu(ref[:, :H]) * ref[:, H:]
        got = out[e * slab:e * slab + c, :ref.shape[1]].float()
        if not torch.isfinite(got).all():
            return float("inf"), float("inf")
        d = float((got - ref).norm()) ** 2
        r = float(ref.norm()) ** 2
        num += d
        den += r
        worst = max(worst, (d / max(r, 1e-30)) ** 0.5)
    return (num / max(den, 1e-30)) ** 0.5, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-humming", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--wide", action="store_true", help="wide (log-uniform incl. tiny) block scales")
    args = ap.parse_args()

    t0 = time.time()
    moe_fp4._ext()
    print(f"compile/load: {time.time() - t0:.1f}s", flush=True)

    gen = torch.Generator(device=dev).manual_seed(1)
    W = {}
    for name, (n, k) in {"w13": (8192, 4096), "w2": (4096, 4096)}.items():
        w, sf, gs = random_weights(n, k, generator=gen, device=dev, leading=(E,), wide_scales=args.wide)
        gs = gs * torch.exp(torch.randn(E, device=dev, generator=gen))
        if name == "w13":  # a few zero / subnormal block scales
            sfu = sf.view(torch.uint8)
            sfu[0, :16, :8] = 0
            sfu[1, 5:9, 3:40] = 3
        W[name] = (w, sf, gs)
    t0 = time.time()
    banks = {"w13": moe_fp4.prepare_bank(*W["w13"]), "w2": moe_fp4.prepare_bank(*W["w2"]),
             "w13f": moe_fp4.prepare_bank(*W["w13"], fuse_gate_up=True)}
    torch.cuda.synchronize()
    print(f"prepare: {time.time() - t0:.1f}s", flush=True)
    hb = {}
    if not args.no_humming:
        hb = {"w13": humming_prepare(*W["w13"]), "w2": humming_prepare(*W["w2"])}
    deq_cache = {}

    def wd_fn(name):
        def f(e):
            key = (name, e)
            if key not in deq_cache:
                if len(deq_cache) > 40:
                    deq_cache.clear()
                w, sf, gs = W[name]
                deq_cache[key] = dequantize(w[e], sf[e], gs[e])
            return deq_cache[key]
        return f

    cases = [(753, "real", 0), (3012, "real", 0), (16384, "real", 0), (300, "dirichlet", 1), (300, "edge", 2),
             (1024, "uniform", 3), (200, "dirichlet", 4), (4096, "dirichlet", 5), (2048, "edge", 6)]
    if args.only:
        keep = {int(v) for v in args.only.split(",")}
        cases = [c for c in cases if c[0] in keep]
    rows = []
    for T, kind, seed in cases:
        counts_cpu = make_counts(T, kind, seed)
        counts = counts_cpu.to(dev)
        R = int(counts_cpu.sum())
        slab = T
        x1 = torch.randn(E * slab, 4096, device=dev, dtype=torch.bfloat16)
        x2 = torch.randn(E * slab, 4096, device=dev, dtype=torch.bfloat16)
        h13 = torch.zeros(E * slab, 8192, device=dev, dtype=torch.bfloat16)
        inter = torch.zeros(E * slab, 4096, device=dev, dtype=torch.bfloat16)
        y = torch.zeros(E * slab, 4096, device=dev, dtype=torch.bfloat16)
        res = {"T": T, "kind": kind, "routes": R}
        f13 = lambda: moe_fp4.run(banks["w13"], x1, counts, h13, slab)
        f13f = lambda: moe_fp4.run_gate_up_silu(banks["w13f"], x1, counts, inter, slab)
        f2 = lambda: moe_fp4.run(banks["w2"], x2, counts, y, slab)
        if not args.no_check:
            f13(); f13f(); f2()
            torch.cuda.synchronize()
            res["err13"] = relerr_all(h13, x1, counts_cpu, slab, wd_fn("w13"))
            res["err13f"] = relerr_all(inter, x1, counts_cpu, slab, wd_fn("w13"), fused=True)
            res["err2"] = relerr_all(y, x2, counts_cpu, slab, wd_fn("w2"))
        res["t13"] = timeit(f13, args.reps)
        res["t13f"] = timeit(f13f, args.reps)
        res["t2"] = timeit(f2, args.reps)
        if hb:
            h13h = torch.zeros_like(h13)
            yh = torch.zeros_like(y)
            g13 = humming_split(hb["w13"], x1, h13h, counts, T)
            g2 = humming_split(hb["w2"], x2, yh, counts, T)
            if not args.no_check:
                g13(); g2()
                torch.cuda.synchronize()
                res["herr13"] = relerr_all(h13h, x1, counts_cpu, slab, wd_fn("w13"))
            res["h13"] = timeit(g13, args.reps)
            res["h2"] = timeit(g2, args.reps)
        rows.append(res)
        line = f"T={T:6d} {kind:9s} routes={R:6d} | ours gate/up {res['t13']:8.1f} us  fused {res['t13f']:8.1f}  down {res['t2']:8.1f}"
        if hb:
            line += f" | humming gate/up {res['h13']:8.1f} down {res['h2']:8.1f}"
            line += f" | speedup (fused+down vs h) {(res['h13'] + res['h2']) / (res['t13f'] + res['t2']):.2f}x"
        if "err13" in res:
            line += f" | relerr13 {res['err13'][0]:.2e}/{res['err13'][1]:.2e} f {res['err13f'][0]:.2e}/{res['err13f'][1]:.2e} 2 {res['err2'][0]:.2e}/{res['err2'][1]:.2e}"
            if "herr13" in res:
                line += f" h13 {res['herr13'][0]:.2e}"
        print(line, flush=True)
        del x1, x2, h13, inter, y
        torch.cuda.empty_cache()
    out = os.path.join(HERE, "results.json")
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
