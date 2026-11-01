"""Hybrid expert GEMM: moe256 (single-variant custom kernel) on full 256-row chunks + humming on each expert's
first (count % 256) rows, vs the current humming big/small split. Judge-observed count distributions."""
import os, sys, json, torch
sys.path.insert(0, "muse_prefill_layer"); sys.path.insert(0, "tools/prefill/moe_kernel")
os.environ["MOE_FP4_BUILD"] = "/scratch-nvme/okc/moe_kernel/hy_build"
import moe256 as M
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
from nvfp4 import dequantize
DISTS = {
    753: [464, 397, 207, 139, 87, 85, 15, 9, 8, 6, 4, 2, 1],
    3012: [2086, 1909, 467, 400, 290, 270, 48, 43, 34, 26, 20, 6, 5, 4],
    16384: [10692, 8996, 4134, 2597, 1646, 1400, 813, 606, 395, 234, 174, 56, 6, 4, 2, 1, 1, 1],
}
def cfg(bm, bk=64):
    return dict(block_shape=(bm, 256, bk), warp_shape=(bm, 32, 64), use_stream_k=True, use_f16_accum=False, num_stages=4,
                use_warp_spec=True, use_tma=True, use_mbarrier=True)
cc = {"gemm_type": "grouped_masked"}
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, (w, sf, gs) = make_layer(n, k)
    bank = M.prepare_bank(w, sf, gs)
    for T, c in DISTS.items():
        counts = (c + [0] * 32)[:32]
        perm = torch.randperm(32, generator=torch.Generator().manual_seed(T))
        cnt = torch.tensor(counts, dtype=torch.int32)[perm].to(dev)
        x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16)
        out = torch.zeros(32 * T, n, device=dev, dtype=torch.bfloat16)
        R = int(cnt.sum())
        def hum(cn, c_):
            layer(x, outputs=out, expert_layout=cn, compute_config=cc, tuning_config=c_, valid_shape_m=R)
        big = torch.where(cnt > 32, cnt, 0).int(); small = torch.where(cnt > 32, 0, cnt).int()
        bmb = 160 if T <= 1024 else 176
        res = {}
        res["humming split"] = timeit(lambda: (hum(big, cfg(bmb)), hum(small, cfg(16, 128))), reps=10)
        head = (cnt % 256).int()
        hb = torch.where(head > 32, head, 0).int(); hs = torch.where(head > 32, 0, head).int()
        res["hybrid: moe256 + humming head split"] = timeit(lambda: (M.run(bank, x, cnt, out, T), hum(hb, cfg(bmb)), hum(hs, cfg(16, 128))), reps=10)
        res["hybrid: moe256 + humming head single bm64"] = timeit(lambda: (M.run(bank, x, cnt, out, T), hum(head, cfg(64))), reps=10)
        res["moe256 only (full chunks)"] = timeit(lambda: M.run(bank, x, cnt, out, T), reps=10)
        # correctness of the hybrid
        out.zero_(); M.run(bank, x, cnt, out, T); hum(hb, cfg(bmb)); hum(hs, cfg(16, 128)); torch.cuda.synchronize()
        worst = 0.0
        for e in range(32):
            ce = int(cnt[e])
            if ce == 0: continue
            ref = x[e * T:e * T + ce].float() @ dequantize(w[e], sf[e], gs[e]).T
            worst = max(worst, float((out[e * T:e * T + ce].float() - ref).norm() / ref.norm()))
        print(f"N={n} T={T} routes={R} worst-expert relerr {worst:.2e}: " + "; ".join(f"{kk} {v:.1f}" for kk, v in res.items()), flush=True)
