"""humming split launches: sequential vs concurrent on two streams with partitioned SMs."""
import sys, os
sys.argv = ["x", "none"]
exec(open("tools/prefill/bench_humming.py").read().split("mode = sys.argv")[0])
DISTS = {
    753: [528, 221, 152, 85, 68, 52, 36, 10, 3, 3, 2, 1, 1, 1, 1] + [0] * 17,
    3012: [2001, 829, 474, 352, 316, 212, 147, 30, 12, 9, 4, 3, 2, 2, 2, 1] + [0] * 16,
    16384: [11956, 4558, 1896, 1807, 1718, 922, 461, 146, 15, 11, 9, 5, 3, 1] + [0] * 18,
}
def cfg(bm, bk=64, sms=0):
    c = dict(block_shape=(bm, 256, bk), warp_shape=(bm, 32, 64), use_stream_k=True, use_f16_accum=False, num_stages=4,
             use_warp_spec=True, use_tma=True, use_mbarrier=True)
    if sms: c["num_sms"] = sms
    return c
cc = {"gemm_type": "grouped_masked"}
side = torch.cuda.Stream()
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, _ = make_layer(n, k)
    locks2 = torch.zeros(1024, dtype=torch.int32, device=dev)
    for T, counts in DISTS.items():
        perm = torch.randperm(32, generator=torch.Generator().manual_seed(T))
        cnt = torch.tensor(counts, dtype=torch.int32)[perm].to(dev)
        x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16)
        out = torch.empty(32 * T, n, device=dev, dtype=torch.bfloat16)
        R = int(cnt.sum())
        big = torch.where(cnt > 64, cnt, 0).int(); small = torch.where(cnt <= 64, cnt, 0).int()
        res = []
        def seq():
            layer(x, outputs=out, expert_layout=big, compute_config=cc, tuning_config=cfg(176), valid_shape_m=R)
            layer(x, outputs=out, expert_layout=small, compute_config=cc, tuning_config=cfg(32, 128), valid_shape_m=R)
        res.append((timeit(seq, reps=10), "sequential"))
        from humming import ops
        def run_on(cnts, c, locks):
            t = layer
            return ops.humming_gemm(layer_config=t.humming_config.to_str(), compute_config=json.dumps(cc), tuning_config=json.dumps(c),
                                    inputs=x, weight=t.weight, weight_scale=t.weight_scale, weight_scale_2=getattr(t, "weight_scale_2", None),
                                    outputs=out, expert_layout=cnts, locks=locks, top_k=1, valid_shape_m=R)
        for sb in [100, 108, 116, 124]:
            ss = 132 - sb
            def conc():
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                run_on(big, cfg(176, 64, sb), layer.locks)
                with torch.cuda.stream(side):
                    run_on(small, cfg(32, 128, ss), locks2)
                main.wait_stream(side)
            res.append((timeit(conc, reps=10), f"concurrent big{sb}/small{ss}"))
        res.sort()
        print(f"N={n} T={T}:", [(round(t, 1), s) for t, s in res[:5]], flush=True)
