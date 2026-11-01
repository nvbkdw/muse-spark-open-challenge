"""humming big experts (> thr rows) + custom kernel (moe_fp4, 16 variants) for small experts, vs humming split."""
import os, sys, torch
sys.path.insert(0, "muse_prefill_layer"); sys.path.insert(0, "tools/prefill/moe_kernel")
os.environ["MOE_FP4_BUILD"] = "/scratch-nvme/okc/moe_kernel/hy_build"
import moe_fp4 as F
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
DISTS = {
    753: [464, 397, 207, 139, 87, 85, 15, 9, 8, 6, 4, 2, 1],
    753.1: [327, 305, 173, 132, 122, 71, 61, 43, 30, 16, 7, 5, 2, 1, 1],
    753.2: [523, 470, 149, 77, 77, 38, 14, 11, 11, 10, 9, 3, 2],
    3012: [2086, 1909, 467, 400, 290, 270, 48, 43, 34, 26, 20, 6, 5, 4],
    3012.1: [1264, 1167, 776, 571, 475, 327, 305, 143, 129, 36, 11, 10, 3, 3, 1, 1],
}
def cfg(bm, bk=64):
    return dict(block_shape=(bm, 256, bk), warp_shape=(bm, 32, 64), use_stream_k=True, use_f16_accum=False, num_stages=4,
                use_warp_spec=True, use_tma=True, use_mbarrier=True)
cc = {"gemm_type": "grouped_masked"}
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, (w, sf, gs) = make_layer(n, k)
    bank = F.prepare_bank(w, sf, gs)
    for Tk, c in DISTS.items():
        T = int(Tk)
        counts = (c + [0] * 32)[:32]
        perm = torch.randperm(32, generator=torch.Generator().manual_seed(T))
        cnt = torch.tensor(counts, dtype=torch.int32)[perm].to(dev)
        x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16)
        out = torch.zeros(32 * T, n, device=dev, dtype=torch.bfloat16)
        R = int(cnt.sum())
        hum = lambda cn, c_: layer(x, outputs=out, expert_layout=cn, compute_config=cc, tuning_config=c_, valid_shape_m=R)
        bmb = 160 if T <= 1024 else 176
        res = {}
        for thr in (32, 64, 128):
            big = torch.where(cnt > thr, cnt, 0).int(); small = torch.where(cnt > thr, 0, cnt).int()
            if thr == 32:
                res["humming split"] = timeit(lambda: (hum(big, cfg(bmb)), hum(small, cfg(16, 128))), reps=10)
                res["humming big only"] = timeit(lambda: hum(big, cfg(bmb)), reps=10)
                res["humming small only"] = timeit(lambda: hum(small, cfg(16, 128)), reps=10)
                res["custom small only"] = timeit(lambda: F.run(bank, x, small, out, T), reps=10)
            res[f"hum big + custom small thr{thr}"] = timeit(lambda: (hum(big, cfg(bmb)), F.run(bank, x, small, out, T)), reps=10)
        print(f"N={n} T={Tk} routes={R}: " + "; ".join(f"{kk} {v:.1f}" for kk, v in res.items()), flush=True)
