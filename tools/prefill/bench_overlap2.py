import sys, os
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
from humming import ops
DISTS = {753: [464, 397, 207, 139, 87, 85, 15, 9, 8, 6, 4, 2, 1], 3012: [2086, 1909, 467, 400, 290, 270, 48, 43, 34, 26, 20, 6, 5, 4]}
def cfg(bm, bk=64, sms=0):
    c = dict(block_shape=(bm, 256, bk), warp_shape=(bm, 32, 64), use_stream_k=True, use_f16_accum=False, num_stages=4, use_warp_spec=True, use_tma=True, use_mbarrier=True)
    if sms: c["num_sms"] = sms
    return c
cc = json.dumps({"gemm_type": "grouped_masked"})
side = torch.cuda.Stream()
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, _ = make_layer(n, k)
    locks2 = torch.zeros(1024, dtype=torch.int32, device=dev)
    for T, c in DISTS.items():
        cnt = torch.tensor((c + [0] * 32)[:32], dtype=torch.int32)[torch.randperm(32, generator=torch.Generator().manual_seed(T))].to(dev)
        x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16); out = torch.empty(32 * T, n, device=dev, dtype=torch.bfloat16)
        R = int(cnt.sum()); bmb = 160 if T <= 1024 else 176
        big = torch.where(cnt > 32, cnt, 0).int(); small = torch.where(cnt > 32, 0, cnt).int()
        run = lambda cn, cf, lk: ops.humming_gemm(layer_config=layer.humming_config.to_str(), compute_config=cc, tuning_config=json.dumps(cf), inputs=x, weight=layer.weight, weight_scale=layer.weight_scale, weight_scale_2=layer.weight_scale_2, outputs=out, expert_layout=cn, locks=lk, top_k=1, valid_shape_m=R)
        res = {"seq big,small": timeit(lambda: (run(big, cfg(bmb), layer.locks), run(small, cfg(16, 128), locks2)), 10),
               "seq small,big": timeit(lambda: (run(small, cfg(16, 128), locks2), run(big, cfg(bmb), layer.locks)), 10)}
        for first in ("big", "small"):
            for ss in (0, 66, 33):
                def conc():
                    main = torch.cuda.current_stream(); side.wait_stream(main)
                    if first == "big":
                        run(big, cfg(bmb), layer.locks)
                        with torch.cuda.stream(side): run(small, cfg(16, 128, ss), locks2)
                    else:
                        with torch.cuda.stream(side): run(small, cfg(16, 128, ss), locks2)
                        run(big, cfg(bmb), layer.locks)
                    main.wait_stream(side)
                res[f"conc {first}-first small_sms={ss or 'all'}"] = timeit(conc, 10)
        print(f"N={n} T={T}: " + "; ".join(f"{kk} {v:.1f}" for kk, v in sorted(res.items(), key=lambda kv: kv[1])), flush=True)
