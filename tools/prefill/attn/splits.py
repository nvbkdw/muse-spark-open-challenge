import sys, torch, triton
from common import *
import attn_impl as A
for name, B, ctx, L, w, p in SHAPES[:2]:
    c = make_case(B, ctx, L, w, p)
    Sp = triton.cdiv(c["page_table"].shape[1] * p, 128) * 128
    Kc, Vt = A.gather_kv(c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], B, Sp)
    res = []
    for ns in [1, 2, 3, 4, 6, 8]:
        t = timeit(lambda: A.custom_attention(c["q"], c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], L, SCALE, c["k_scale"], c["v_scale"], num_splits=ns))
        res.append(f"ns{ns}:{t:.0f}")
    print(name, " ".join(res), flush=True)
