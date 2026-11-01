"""FA3 at the prefill shapes: paged (page_size 1, scattered) vs contiguous KV (page_size = whole seq)."""
import sys, os, torch
sys.path.insert(0, "muse_prefill_layer")
from sgl_kernel.flash_attn import flash_attn_with_kvcache
torch.manual_seed(0)
dev = "cuda"
def timeit(fn, reps=10):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    fl = torch.empty(512 << 20, dtype=torch.uint8, device=dev); ts = []
    for _ in range(reps):
        fl.fill_(1); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
for (B, ctx, L, window) in [(1, 80000, 753, 0), (4, 80000, 753, 0), (1, 32768, 16384, 0), (1, 65536, 16384, 0), (1, 32768, 16384, 2048), (1, 80000, 753, 2048)]:
    S = ctx + L
    T = B * L
    q = torch.randn(T, 16, 64, device=dev).to(torch.float8_e4m3fn)
    pages = B * S
    kc = torch.randn(pages, 1, 2, 64, device=dev).to(torch.float8_e4m3fn)
    vc = torch.randn(pages, 1, 2, 64, device=dev).to(torch.float8_e4m3fn)
    perm = torch.randperm(pages, device=dev).int().view(B, S)
    seqlens = torch.full((B,), S, dtype=torch.int32, device=dev)
    cu = torch.arange(0, T + 1, L, dtype=torch.int32, device=dev)
    ws = (window - 1, 0) if window else (-1, -1)
    f1 = lambda: flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, page_table=perm, cache_seqlens=seqlens, cu_seqlens_q=cu,
                                         max_seqlen_q=L, softmax_scale=0.125 * 5.47, causal=True, window_size=ws)
    t1 = timeit(f1)
    # contiguous: [B, S, 2, 64]
    kcc = torch.randn(B, S, 2, 64, device=dev).to(torch.float8_e4m3fn)
    vcc = torch.randn(B, S, 2, 64, device=dev).to(torch.float8_e4m3fn)
    f2 = lambda: flash_attn_with_kvcache(q=q.view(B, L, 16, 64), k_cache=kcc, v_cache=vcc, cache_seqlens=seqlens,
                                         softmax_scale=0.125 * 5.47, causal=True, window_size=ws)
    t2 = timeit(f2)
    # paged with page 64 (TMA-able?)
    P = 128
    np_ = (S + P - 1) // P
    kcp = torch.randn(B * np_, P, 2, 64, device=dev).to(torch.float8_e4m3fn)
    vcp = torch.randn(B * np_, P, 2, 64, device=dev).to(torch.float8_e4m3fn)
    pt = torch.randperm(B * np_, device=dev).int().view(B, np_)
    f3 = lambda: flash_attn_with_kvcache(q=q, k_cache=kcp, v_cache=vcp, page_table=pt, cache_seqlens=seqlens, cu_seqlens_q=cu,
                                         max_seqlen_q=L, softmax_scale=0.125 * 5.47, causal=True, window_size=ws)
    t3 = timeit(f3)
    att = min(ctx, window) if window else ctx
    fl = 4 * T * (att + L / 2) * 16 * 64
    print(f"B{B} ctx{ctx} L{L} win{window}: paged1 {t1:.1f}us ({fl/t1/1e6:.0f} TF/s)  contiguous {t2:.1f}us ({fl/t2/1e6:.0f})  paged{P} {t3:.1f}us", flush=True)
