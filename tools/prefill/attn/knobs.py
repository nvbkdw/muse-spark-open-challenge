import sys, torch
from common import *
from sgl_kernel.flash_attn import flash_attn_with_kvcache
def fa3(c, **kw):
    w = c["window"]
    return flash_attn_with_kvcache(q=c["q"], k_cache=c["k_cache"], v_cache=c["v_cache"], page_table=c["page_table"],
        cache_seqlens=c["cache_seqlens"], cu_seqlens_q=c["cu_seqlens_q"], max_seqlen_q=c["max_seqlen_q"],
        softmax_scale=SCALE, causal=True, window_size=(w - 1, 0) if w else (-1, -1), **kw)
for name, B, ctx, L, w, p in SHAPES[:5]:
    c = make_case(B, ctx, L, w, p)
    res = []
    for kw in [{}, dict(pack_gqa=True), dict(pack_gqa=False), dict(num_splits=1), dict(num_splits=2), dict(num_splits=4), dict(pack_gqa=True, num_splits=1), dict(pack_gqa=False,num_splits=1)]:
        t = timeit(lambda: fa3(c, **kw))
        res.append(f"{kw}:{t:.0f}")
    print(name, f"{flops(c)/1e12:.3f}TF", " | ".join(res), flush=True)
    del c; torch.cuda.empty_cache()
