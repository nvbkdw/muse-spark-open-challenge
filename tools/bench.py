"""Local judge-like benchmark for muse_bf16_proj.

Mimics the OKC judge: every shape is captured as a CUDA graph for both the
submission and the SGLang-equivalent baseline; each timed replay starts with a
cold L2 (a 512 MB buffer is read first); replays alternate in ABBA order; a
shape's speedup is the median of paired ratios; load time = sum(weight*time);
problem speedup = geomean over loads of baseline/sub.  An empty-graph replay
cost is measured and subtracted from both sides like the judge does.

usage: python tools/bench.py [submission.py] [--reps N] [--only layer] [--m 1,16]
"""
import argparse, importlib.util, math, os, statistics, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.join(HERE, "..", "muse_bf16_proj")
sys.path.insert(0, PROB)
import reference  # noqa: E402

LOADS = {"decode 1": 1, "decode 16": 16, "decode 32": 32, "verify 1": 8, "verify 16": 128, "verify 32": 256}
LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baseline_module():
    import torch.nn.functional as F

    class B:
        @staticmethod
        def prepare(weights):
            st = {k: v.contiguous() for k, v in weights.items()}
            r = weights["router"]
            hi = r.to(torch.bfloat16)
            lo = (r - hi.float()).to(torch.bfloat16)
            st["router_hi_t"] = hi.t().contiguous()
            st["router_lo_t"] = lo.t().contiguous()
            return st

        @staticmethod
        def kernel_fn(data, st):
            x, layer = data["x"], data["config"]["layer"]
            if layer != "router":
                return F.linear(x, st[layer][: data["config"]["n"]])
            if x.shape[0] <= 8:
                return F.linear(x.float(), st["router"])
            acc = torch.mm(x, st["router_hi_t"], out_dtype=torch.float32)
            acc += torch.mm(x, st["router_lo_t"], out_dtype=torch.float32)
            return acc
    return B


def capture(fn, data, st):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(data, st)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(data, st)
    return g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub", nargs="?", default=os.path.join(PROB, "submission.py"))
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--only", default=None)
    ap.add_argument("--m", default=None)
    ap.add_argument("--nobase", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    weights = reference.generate_weights(reference.WEIGHTS_SEED)
    sub = load_mod(a.sub, "sub")
    base = baseline_module()
    st_s = sub.prepare({k: v.clone() for k, v in weights.items()})
    st_b = base.prepare(weights)
    flush = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    junk = torch.empty(1, device="cuda")
    # empty-graph fixed cost
    eg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(eg):
        junk.add_(0)
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def time_graph(g):
        torch.sum(flush, out=junk[0]) if False else flush.sum()
        st.record(); g.replay(); en.record(); en.synchronize()
        return st.elapsed_time(en) * 1000.0

    fixed = statistics.median([time_graph(eg) for _ in range(50)])
    ms = [int(v) for v in a.m.split(",")] if a.m else None
    res = {}
    for load, m in LOADS.items():
        if ms and m not in ms:
            continue
        for layer in LAYERS:
            if a.only and layer not in a.only.split(","):
                continue
            data = reference.generate_input(m, layer, seed=1234 + m)
            gs, outs = capture(sub.kernel_fn, data, st_s)
            gs.replay(); torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, outs, weights)
            if not a.nobase:
                gb, outb = capture(base.kernel_fn, data, st_b)
            ts, tb, ratios = [], [], []
            for r in range(a.reps):
                if a.nobase:
                    ts.append(time_graph(gs) - fixed); continue
                if r % 2 == 0:
                    s1 = time_graph(gs); b1 = time_graph(gb); b2 = time_graph(gb); s2 = time_graph(gs)
                else:
                    b1 = time_graph(gb); s1 = time_graph(gs); s2 = time_graph(gs); b2 = time_graph(gb)
                for s_, b_ in ((s1, b1), (s2, b2)):
                    s_ -= fixed; b_ -= fixed
                    ts.append(s_); tb.append(b_); ratios.append(b_ / s_)
            tsm = statistics.median(ts)
            if a.nobase:
                print(f"{load:10s} m={m:4d} {layer:9s} sub {tsm:7.2f}us  {'OK' if ok else 'FAIL ' + msg}", flush=True)
                res[(load, layer)] = (tsm, tsm)
                continue
            tbm = statistics.median(tb)
            sp = statistics.median(ratios)
            res[(load, layer)] = (tbm, tbm / sp)
            print(f"{load:10s} m={m:4d} {layer:9s} base {tbm:7.2f}us sub {tbm/sp:7.2f}us  x{sp:5.3f}  {'OK' if ok else 'FAIL ' + msg}", flush=True)
            del gs
            if not a.nobase:
                del gb
    if a.nobase:
        return
    sps = []
    for load in LOADS:
        items = [v for (l, _), v in res.items() if l == load]
        if len(items) == 5:
            b = sum(i[0] for i in items); s = sum(i[1] for i in items)
            sps.append(b / s)
            print(f"LOAD {load:10s} base {b:7.2f} sub {s:7.2f} x{b/s:5.3f}")
    if sps:
        print(f"GEOMEAN x{math.exp(sum(map(math.log, sps)) / len(sps)):.4f}  (fixed replay cost {fixed:.2f}us)")


if __name__ == "__main__":
    main()
