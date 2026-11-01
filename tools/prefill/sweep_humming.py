import sys, itertools, json
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
n, k = int(sys.argv_n) if False else None, None
import os
N, K = map(int, os.environ.get("NK", "8192,4096").split(","))
ROUTES = [int(x) for x in os.environ.get("ROUTES", "1500,5600").split(",")]
layer, _ = make_layer(N, K)
torch.manual_seed(0)
cases = []
for routes in ROUTES:
    p = torch.distributions.Dirichlet(torch.full((E,), 2.0)).sample().to(dev)
    ids = torch.multinomial(p, routes, replacement=True)
    counts = torch.bincount(ids, minlength=E)
    layout = torch.nn.functional.pad(counts.cumsum(0), (1, 0)).long()
    x = torch.randn(routes, K, device=dev, dtype=torch.bfloat16)
    out = torch.empty(routes, N, device=dev, dtype=torch.bfloat16)
    cases.append((routes, layout, x, out, int(counts.max())))
cc = {"gemm_type": "grouped_contiguous"}
grid = []
for bm in [int(v) for v in os.environ.get("BMS", "16,32,48,64,96,128").split(",")]:
    for bn in [128, 256]:
        for bk in [64, 128, 256]:
            for st in [4, 6]:
                for sk in [True, False]:
                    grid.append(dict(block_shape=(bm, bn, bk), warp_shape=(bm, 32, 64), use_stream_k=sk, use_f16_accum=False,
                                     num_stages=st, use_warp_spec=True, use_tma=True, use_mbarrier=True))
res = {}
for cfg in grid:
    for (routes, layout, x, out, mx) in cases:
        try:
            fn = lambda: layer(x, outputs=out, expert_layout=layout, compute_config=cc, tuning_config=cfg)
            t = timeit(fn, reps=10)
        except Exception as ex:
            t = None
        res.setdefault(routes, []).append((t, cfg))
        if t: print(routes, f"{t:.1f}", cfg["block_shape"], cfg["num_stages"], cfg["use_stream_k"], flush=True)
for routes, lst in res.items():
    lst = sorted([r for r in lst if r[0]], key=lambda r: r[0])
    print("BEST", routes, [(round(t, 1), c["block_shape"], c["num_stages"], c["use_stream_k"]) for t, c in lst[:5]])
