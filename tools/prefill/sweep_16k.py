import sys, os
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
c = [10692, 8996, 4134, 2597, 1646, 1400, 813, 606, 395, 234, 174, 56, 6, 4, 2, 1, 1, 1]
T = 16384
cc = {"gemm_type": "grouped_masked"}
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, _ = make_layer(n, k)
    cnt = torch.tensor((c + [0] * 32)[:32], dtype=torch.int32)[torch.randperm(32, generator=torch.Generator().manual_seed(3))].to(dev)
    x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16); out = torch.empty(32 * T, n, device=dev, dtype=torch.bfloat16)
    R = int(cnt.sum()); res = []
    for bm in (128, 144, 160, 176):
        for st in (3, 4, 5):
            for sk in (True, False):
                for bn in (256, 128):
                    cf = dict(block_shape=(bm, bn, 64), warp_shape=(bm, 32, 64), use_stream_k=sk, use_f16_accum=False, num_stages=st, use_warp_spec=True, use_tma=True, use_mbarrier=True)
                    try:
                        t = timeit(lambda: layer(x, outputs=out, expert_layout=cnt, compute_config=cc, tuning_config=cf, valid_shape_m=R), reps=5)
                    except Exception:
                        continue
                    res.append((t, f"bm{bm} bn{bn} st{st} sk{int(sk)}"))
    res.sort()
    print(f"N={n} T={T} routes={R}: " + "; ".join(f"{t:.0f} {s} ({2*R*n*k/t/1e6:.0f}TF)" for t, s in res[:6]), flush=True)
