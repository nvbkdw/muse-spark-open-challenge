"""humming masked MoE with realistic skewed counts: single launch vs big/small split launches."""
import sys, os, itertools
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
DISTS = {
    753: [528, 221, 152, 85, 68, 52, 36, 10, 3, 3, 2, 1, 1, 1, 1] + [0] * 17,
    3012: [2001, 829, 474, 352, 316, 212, 147, 30, 12, 9, 4, 3, 2, 2, 2, 1] + [0] * 16,
}
def cfg(bm, bk=64, st=4, sk=True):
    return dict(block_shape=(bm, 256, bk), warp_shape=(bm, 32, 64), use_stream_k=sk, use_f16_accum=False, num_stages=st,
                use_warp_spec=True, use_tma=True, use_mbarrier=True)
cc = {"gemm_type": "grouped_masked"}
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, _ = make_layer(n, k)
    for T, counts in DISTS.items():
        g = torch.Generator().manual_seed(T)
        perm = torch.randperm(32, generator=g)
        cnt = torch.tensor(counts, dtype=torch.int32)[perm].to(dev)
        x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16)
        out = torch.empty(32 * T, n, device=dev, dtype=torch.bfloat16)
        R = int(cnt.sum())
        best = []
        for bm in [48, 64, 96, 128, 176]:
            for bk in ([128, 64] if bm <= 64 else [64]):
                c = cfg(bm, bk)
                t = timeit(lambda: layer(x, outputs=out, expert_layout=cnt, compute_config=cc, tuning_config=c, valid_shape_m=R), reps=10)
                best.append((t, f"single bm{bm} bk{bk}"))
        for thr in [32, 64, 128]:
            big = torch.where(cnt > thr, cnt, 0).int(); small = torch.where(cnt <= thr, cnt, 0).int()
            for bmb in [128, 176]:
                for bms in [16, 32, 48, 64]:
                    if bms > thr: continue
                    cb, cs = cfg(bmb), cfg(bms, 128)
                    def fn():
                        layer(x, outputs=out, expert_layout=big, compute_config=cc, tuning_config=cb, valid_shape_m=R)
                        layer(x, outputs=out, expert_layout=small, compute_config=cc, tuning_config=cs, valid_shape_m=R)
                    t = timeit(fn, reps=10)
                    best.append((t, f"split thr{thr} big{bmb} small{bms}"))
        best.sort()
        print(f"N={n} T={T} routes={R}:", [(round(t, 1), s) for t, s in best[:6]], flush=True)
