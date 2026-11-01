"""Custom hybrid: moe256 (N=256 only) on full chunks + moe_fp4 (16 variants) on each expert's first c%256 rows."""
import os, sys, torch
sys.path.insert(0, "muse_prefill_layer"); sys.path.insert(0, "tools/prefill/moe_kernel")
os.environ["MOE_FP4_BUILD"] = "/scratch-nvme/okc/moe_kernel/hy_build"
import moe256 as M
import moe_fp4 as F
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
from nvfp4 import dequantize
DISTS = {
    753: [464, 397, 207, 139, 87, 85, 15, 9, 8, 6, 4, 2, 1],
    753.1: [327, 305, 173, 132, 122, 71, 61, 43, 30, 16, 7, 5, 2, 1, 1],
    3012: [2086, 1909, 467, 400, 290, 270, 48, 43, 34, 26, 20, 6, 5, 4],
    3012.1: [1264, 1167, 776, 571, 475, 327, 305, 143, 129, 36, 11, 10, 3, 3, 1, 1],
    300: [170, 163, 92, 43, 36, 35, 19, 8, 8, 7, 3, 1],
}
def cfg(bm, bk=64):
    return dict(block_shape=(bm, 256, bk), warp_shape=(bm, 32, 64), use_stream_k=True, use_f16_accum=False, num_stages=4,
                use_warp_spec=True, use_tma=True, use_mbarrier=True)
cc = {"gemm_type": "grouped_masked"}
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, (w, sf, gs) = make_layer(n, k)
    bank = M.prepare_bank(w, sf, gs)
    for Tk, c in DISTS.items():
        T = int(Tk)
        counts = (c + [0] * 32)[:32]
        perm = torch.randperm(32, generator=torch.Generator().manual_seed(T))
        cnt = torch.tensor(counts, dtype=torch.int32)[perm].to(dev)
        x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16)
        out = torch.zeros(32 * T, n, device=dev, dtype=torch.bfloat16)
        R = int(cnt.sum())
        hum = lambda cn, c_: layer(x, outputs=out, expert_layout=cn, compute_config=cc, tuning_config=c_, valid_shape_m=R)
        big = torch.where(cnt > 32, cnt, 0).int(); small = torch.where(cnt > 32, 0, cnt).int()
        bmb = 160 if T <= 1024 else 176
        head = (cnt % 256).int()
        res = {}
        res["humming split"] = timeit(lambda: (hum(big, cfg(bmb)), hum(small, cfg(16, 128))), reps=10)
        res["custom16 all"] = timeit(lambda: F.run(bank, x, cnt, out, T), reps=10)
        res["moe256 + custom16 heads"] = timeit(lambda: (M.run(bank, x, cnt, out, T), F.run(bank, x, head, out, T)), reps=10)
        out.zero_(); M.run(bank, x, cnt, out, T); F.run(bank, x, head, out, T); torch.cuda.synchronize()
        worst = 0.0
        for e in range(32):
            ce = int(cnt[e])
            if ce == 0: continue
            ref = x[e * T:e * T + ce].float() @ dequantize(w[e], sf[e], gs[e]).T
            worst = max(worst, float((out[e * T:e * T + ce].float() - ref).norm() / ref.norm()))
        print(f"N={n} T={Tk} routes={R} worst relerr {worst:.2e}: " + "; ".join(f"{kk} {v:.1f}" for kk, v in res.items()), flush=True)
