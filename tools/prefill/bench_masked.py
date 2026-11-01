import sys, os
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
from nvfp4 import dequantize
T = int(os.environ.get("T", "16384"))
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, (w, sf, gs) = make_layer(n, k)
    for routes in [int(r) for r in os.environ.get("ROUTES", "1350,5570,30500").split(",")]:
        torch.manual_seed(routes)
        p = torch.distributions.Dirichlet(torch.full((E,), 1.0)).sample().to(dev)
        ids = torch.multinomial(p, routes, replacement=True)
        counts = torch.bincount(ids, minlength=E).clamp(max=T)
        slab = T
        x = torch.randn(E * slab, k, device=dev, dtype=torch.bfloat16)
        out = torch.zeros(E * slab, n, device=dev, dtype=torch.bfloat16)
        cnt = counts.int()
        cc = {"gemm_type": "grouped_masked"}
        cfg = None
        if os.environ.get("CFG"):
            import ast; cfg = ast.literal_eval(os.environ["CFG"])
        fn = lambda: layer(x, outputs=out, expert_layout=cnt, compute_config=cc, tuning_config=cfg, valid_shape_m=routes)
        t = timeit(fn)
        fl = 2 * int(counts.sum()) * n * k
        print(f"masked N={n} K={k} routes={int(counts.sum())} max/expert={int(counts.max())}: {t:.1f} us {fl / t / 1e6:.0f} TFLOP/s", flush=True)
        # correctness on 3 experts
        for e in [0, int(counts.argmax()), int(counts.argmin())]:
            c = int(counts[e])
            if c == 0: continue
            wd = dequantize(w[e], sf[e], gs[e])
            ref = x[e * slab: e * slab + c].float() @ wd.T
            got = out[e * slab: e * slab + c].float()
            print(f"   expert {e} rows {c} relerr {float((got - ref).norm() / ref.norm()):.2e}")
