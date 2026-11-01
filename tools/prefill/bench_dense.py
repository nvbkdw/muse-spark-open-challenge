import torch
dev = "cuda"
def timeit(fn, reps=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    fl = torch.empty(512 << 20, dtype=torch.uint8, device=dev); ts = []
    for _ in range(reps):
        fl.fill_(1); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
for T in [753, 3012, 16384]:
    for name, (n, k) in {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (512, 8192)}.items():
        x = torch.randn(T, k, device=dev).bfloat16()
        w = torch.randn(n, k, device=dev).bfloat16()
        wt = w.t().contiguous()
        fl = 2 * T * n * k
        out_dt = torch.float32 if name == "router" else torch.bfloat16
        r = {}
        r["TN x@w.T"] = timeit(lambda: torch.mm(x, w.t(), out_dtype=out_dt) if name == "router" else torch.mm(x, w.t()))
        r["NN x@wt"] = timeit(lambda: torch.mm(x, wt, out_dtype=out_dt) if name == "router" else torch.mm(x, wt))
        if name != "router":
            out = torch.empty(n, T, device=dev).bfloat16()
            r["swap w@x.T"] = timeit(lambda: torch.mm(w, x.t(), out=out))
        best = min(r, key=r.get)
        print(f"T={T} {name:8s} " + "  ".join(f"{kk}: {v:6.1f}us ({fl / v / 1e6:4.0f})" for kk, v in r.items()) + f"  -> {best}", flush=True)
