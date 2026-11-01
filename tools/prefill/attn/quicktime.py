import sys, torch
from common import *
import attn_impl as A
from sgl_kernel.flash_attn import flash_attn_with_kvcache
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else None
for name, B, ctx, L, w, p in SHAPES[:5]:
    if sel and name not in sel: continue
    c = make_case(B, ctx, L, w, p)
    f1 = lambda: flash_attn_with_kvcache(q=c["q"], k_cache=c["k_cache"], v_cache=c["v_cache"], page_table=c["page_table"],
        cache_seqlens=c["cache_seqlens"], cu_seqlens_q=c["cu_seqlens_q"], max_seqlen_q=L, softmax_scale=SCALE, causal=True)
    f2 = lambda: A.custom_attention(c["q"], c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], L, SCALE, c["k_scale"], c["v_scale"])
    Spad = triton.cdiv(c["page_table"].shape[1] * p, 128) * 128 if False else None
    t1 = timeit(f1); t2 = timeit(f2)
    import triton
    Sp = triton.cdiv(c["page_table"].shape[1] * p, 128) * 128
    tg = timeit(lambda: A.gather_kv(c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], B, Sp))
    fl = flops(c)
    print(f"{name}: fa3 {t1:.0f}us ({fl/t1/1e6:.0f} TF/s)  mine {t2:.0f}us ({fl/t2/1e6:.0f} TF/s)  [gather {tg:.0f}us]  speedup {t1/t2:.3f}", flush=True)
    del c; torch.cuda.empty_cache()
