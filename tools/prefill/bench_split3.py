import sys, os
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
from humming import ops
DISTS = {753: [464, 397, 207, 139, 87, 85, 15, 9, 8, 6, 4, 2, 1], 753.1: [327, 305, 173, 132, 122, 71, 61, 43, 30, 16, 7, 5, 2, 1, 1],
         753.2: [523, 470, 149, 77, 77, 38, 14, 11, 11, 10, 9, 3, 2], 3012: [2086, 1909, 467, 400, 290, 270, 48, 43, 34, 26, 20, 6, 5, 4],
         3012.1: [1264, 1167, 776, 571, 475, 327, 305, 143, 129, 36, 11, 10, 3, 3, 1, 1]}
def cfg(bm, bk=64):
    return dict(block_shape=(bm, 256, bk), warp_shape=(bm, 32, 64), use_stream_k=True, use_f16_accum=False, num_stages=4, use_warp_spec=True, use_tma=True, use_mbarrier=True)
cc = json.dumps({"gemm_type": "grouped_masked"})
sides = [torch.cuda.Stream(), torch.cuda.Stream()]
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, _ = make_layer(n, k)
    locks = [layer.locks, torch.zeros(1024, dtype=torch.int32, device=dev), torch.zeros(1024, dtype=torch.int32, device=dev)]
    for Tk, c in DISTS.items():
        T = int(Tk)
        cnt = torch.tensor((c + [0] * 32)[:32], dtype=torch.int32)[torch.randperm(32, generator=torch.Generator().manual_seed(T))].to(dev)
        x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16); out = torch.empty(32 * T, n, device=dev, dtype=torch.bfloat16)
        R = int(cnt.sum())
        run = lambda cn, cf, i: ops.humming_gemm(layer_config=layer.humming_config.to_str(), compute_config=cc, tuning_config=json.dumps(cf), inputs=x, weight=layer.weight, weight_scale=layer.weight_scale, weight_scale_2=layer.weight_scale_2, outputs=out, expert_layout=cn, locks=locks[i], top_k=1, valid_shape_m=R)
        def multi(parts):
            main = torch.cuda.current_stream()
            for s in sides: s.wait_stream(main)
            for i, (cn, cf) in enumerate(parts):
                if i == 0: run(cn, cf, 0)
                else:
                    with torch.cuda.stream(sides[i - 1]): run(cn, cf, i)
            for s in sides: main.wait_stream(s)
        bmb = 160 if T <= 1024 else 176
        res = {}
        big = torch.where(cnt > 32, cnt, 0).int(); small = torch.where(cnt > 32, 0, cnt).int()
        res["2way (current)"] = timeit(lambda: multi([(big, cfg(bmb)), (small, cfg(16, 128))]), 10)
        for hi in (192, 256, 320):
            for bmm in (64, 80, 96, 112):
                b3 = torch.where(cnt > hi, cnt, 0).int(); m3 = torch.where((cnt > 32) & (cnt <= hi), cnt, 0).int()
                res[f"3way >{hi}:bm{bmb} mid:bm{bmm}"] = timeit(lambda: multi([(b3, cfg(bmb)), (m3, cfg(bmm)), (small, cfg(16, 128))]), 10)
        best = sorted(res.items(), key=lambda kv: kv[1])
        print(f"N={n} T={Tk}: current {res['2way (current)']:.1f} | best: " + "; ".join(f"{k_} {v:.1f}" for k_, v in best[:3]), flush=True)
