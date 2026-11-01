"""Shared helpers for the attention experiments: case generation (reference-style paged FP8 cache)
and CUDA-graph cold-L2 timing (same method as tools/prefill/bench_fa3.py)."""
import torch

FP8 = torch.float8_e4m3fn
SCALE = 0.125 * 5.4730064863875
dev = "cuda"

SHAPES = [  # (name, B, ctx, L, window, page)
    ("full_b1_80k_753", 1, 80000, 753, 0, 1),
    ("full_b4_80k_753", 4, 80000, 753, 0, 1),
    ("full_b1_0_16k", 1, 0, 16384, 0, 1),
    ("full_b1_32k_16k", 1, 32768, 16384, 0, 1),
    ("full_b1_64k_16k", 1, 65536, 16384, 0, 1),
    ("slid_b1_80k_753", 1, 80000, 753, 2048, 1),
    ("slid_b4_80k_753", 4, 80000, 753, 2048, 1),
    ("slid_b1_32k_16k", 1, 32768, 16384, 2048, 1),
    ("full_b2_3000_753_p64", 2, 3000, 753, 0, 64),
    ("full_b3_5000_200_var", 3, 5000, 200, 0, 1),
]


def make_case(B, ctx, L, window, page, seed=0, soft=True, kv_scale=1.0, var_lens=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    if var_lens:
        seq_lens = (ctx * torch.exp(torch.randn(B, generator=g, device=dev) - 0.5)).clamp(16, 4 * ctx).long()
    else:
        seq_lens = torch.full((B,), ctx, dtype=torch.long, device=dev)
    pages_per = (seq_lens + L + page - 1) // page
    longest = 4 * ctx if var_lens else ctx
    max_pages = (longest + L + page - 1) // page
    total = B * max_pages
    perm = torch.randperm(total, generator=g, device=dev).to(torch.int32)
    pt = torch.full((B, max_pages), -1, dtype=torch.int32, device=dev)
    off = torch.cumsum(pages_per, 0) - pages_per
    for b in range(B):
        n, o = int(pages_per[b]), int(off[b])
        pt[b, :n] = perm[o:o + n]
    shape = (total, page, 2, 64)
    k = torch.randn(shape, generator=g, device=dev)
    if soft:
        u = torch.randn(2, 64, generator=g, device=dev)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
        c2 = 0.5 + 0.12 * torch.rand(2, 1, generator=g, device=dev)
        k = c2.sqrt() * u + (1 - c2).sqrt() * k
    kc = (k / kv_scale).to(FP8)
    del k
    vc = (torch.randn(shape, generator=g, device=dev) / kv_scale).to(FP8)
    q = torch.randn(B * L, 16, 64, generator=g, device=dev)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True))
    if soft:  # queries partially aligned with the shared key direction, like the real model
        q = 0.7 * q + 0.7 * u.repeat_interleave(8, 0)[None]
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True))
    q8 = q.to(FP8)
    return dict(q=q8, k_cache=kc, v_cache=vc, page_table=pt,
                cache_seqlens=(seq_lens + L).to(torch.int32),
                cu_seqlens_q=torch.arange(0, B * L + 1, L, dtype=torch.int32, device=dev),
                max_seqlen_q=L, window=window, page=page,
                k_scale=torch.tensor(kv_scale, device=dev), v_scale=torch.tensor(kv_scale, device=dev),
                kv_scale=kv_scale, B=B, L=L, seq_lens=seq_lens)


def timeit(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    fl = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    ts = []
    for _ in range(reps):
        fl.fill_(1)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    del g
    return ts[len(ts) // 2]


def flops(c):
    B, L, w = c["B"], c["L"], c["window"]
    tot = 0.0
    for b in range(B):
        ctx = int(c["seq_lens"][b])
        for _ in range(1):
            if w:
                # approx: each query sees min(pos+1, w) keys
                import numpy as np
                pos = ctx + np.arange(L)
                tot += 4 * 16 * 64 * float(np.minimum(pos + 1, w).sum())
            else:
                tot += 4 * 16 * 64 * L * (ctx + (L + 1) / 2)
    return tot
