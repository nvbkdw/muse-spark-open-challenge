"""Local judge-like harness for muse_prefill_layer (H100 stand-in for the H200 judge).

  python tools/prefill/judge.py [--sub muse_prefill_layer/submission.py] [--cases tests,scored] [--no-check]
                                [--no-time] [--reps 10] [--only IDX,...] [--router-seed N]

Correctness: reference.check_implementation on the collected outputs (eager call, then a CUDA graph
replay on fresh inputs). Timing: CUDA graph of kernel_fn replayed on one input with a >L2 buffer
written just before each replay; median of reps (fixed replay cost not subtracted).
"""
import argparse
import importlib.util
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "muse_prefill_layer"))
import reference as R  # noqa: E402

TESTS = [
    dict(batch=1, ctx=0, layer="sliding", q_len=300, page=1),
    dict(batch=1, ctx=3000, layer="full", q_len=753, page=1),
    dict(batch=2, ctx=2500, layer="sliding", q_len=753, page=1),
    dict(batch=3, ctx=5000, layer="full", q_len=200, dist="outliers"),
    dict(batch=1, ctx=4096, layer="sliding", q_len=4096, page=1),
]
SCORED = [
    dict(batch=1, ctx=80000, layer="sliding", q_len=753, page=1, dist="soft", w=46, load="turn 1", sgl=1079.0),
    dict(batch=1, ctx=80000, layer="full", q_len=753, page=1, dist="soft", w=16, load="turn 1", sgl=1469.7),
    dict(batch=4, ctx=80000, layer="sliding", q_len=753, page=1, dist="soft", w=46, load="turn 4", sgl=3275.2),
    dict(batch=4, ctx=80000, layer="full", q_len=753, page=1, dist="soft", w=16, load="turn 4", sgl=4930.2),
    dict(batch=1, ctx=0, layer="full", q_len=16384, page=1, dist="soft", w=16, load="first turn", sgl=18144.0),
    dict(batch=1, ctx=32768, layer="sliding", q_len=16384, page=1, dist="soft", w=138, load="first turn", sgl=17045.0),
    dict(batch=1, ctx=32768, layer="full", q_len=16384, page=1, dist="soft", w=16, load="first turn", sgl=21887.8),
    dict(batch=1, ctx=65536, layer="full", q_len=16384, page=1, dist="soft", w=16, load="first turn", sgl=24611.6),
]


def load_sub(path):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case_args(c, seed):
    return {k: v for k, v in c.items() if k not in ("w", "load", "sgl")} | {"seed": seed}


def time_graph(fn, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    ts = []
    for _ in range(reps):
        flush.fill_(1)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(ROOT, "muse_prefill_layer", "submission.py"))
    ap.add_argument("--cases", default="tests,scored")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--only", default="")
    ap.add_argument("--router-seed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(0)
    weights = R.generate_weights(R.WEIGHTS_SEED, router_seed=args.router_seed)
    sub = load_sub(args.sub)
    t0 = time.time()
    state = sub.prepare({k: v.clone() for k, v in weights.items()})
    torch.cuda.synchronize()
    print(f"prepare {time.time() - t0:.1f}s", flush=True)

    cases = []
    for name in args.cases.split(","):
        cases += [("test", c) for c in TESTS] if name == "tests" else [("scored", c) for c in SCORED]
    only = {int(i) for i in args.only.split(",") if i}
    results = {}
    ok_all = True
    for idx, (kind, c) in enumerate(cases):
        if only and idx not in only:
            continue
        seed = args.seed + idx
        ca = case_args(c, seed)
        ls = R.case_fields(ca, weights)["local_start"]
        data = R.generate_input(**ca, local_start=ls)
        line = f"[{idx}] {kind} {c.get('load', '')} b{c['batch']} ctx{c['ctx']} {c['layer']} L{c['q_len']}"
        if not args.no_check:
            out = sub.kernel_fn(data, state)
            torch.cuda.synchronize()
            raw = R.collect(data, out)
            ok, msg = R.check_implementation(data, raw, weights)
            ids = out[2].long()
            routes, distinct = R._local_load(ids, ls)
            line += f" routes={routes} distinct={distinct} | {'PASS' if ok else 'FAIL'} {msg}"
            ok_all &= ok
            del out, raw
        if not args.no_time:
            fresh = R.generate_input(**case_args(c, seed + 1000), local_start=ls)
            t, g = time_graph(lambda: sub.kernel_fn(fresh, state), args.reps)
            line += f" | {t:.1f} us"
            if "sgl" in c:
                line += f" (judge SGLang H200 {c['sgl']:.0f} us, ratio {c['sgl'] / t:.2f})"
                results[idx] = (c, t)
            if not args.no_check:
                # the graph replays on fresh inputs too: check the replayed output
                outg = sub.kernel_fn  # noqa: F841
            del g, fresh
        print(line, flush=True)
        del data
        torch.cuda.empty_cache()
    if results and len(results) == len(SCORED):
        loads = {}
        for c, t in results.values():
            loads.setdefault(c["load"], [0.0, 0.0])
            loads[c["load"]][0] += c["w"] * c["sgl"]
            loads[c["load"]][1] += c["w"] * t
        sp = math.prod(v[0] / v[1] for v in loads.values()) ** (1 / len(loads))
        print("per-load ratio (H200 SGLang / local H100):", {k: round(v[0] / v[1], 3) for k, v in loads.items()},
              f"geomean {sp:.3f}")
    print("ALL PASS" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()
