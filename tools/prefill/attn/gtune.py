import torch, triton
from common import *
import attn_impl as A
c = make_case(4, 80000, 753, 0, 1)
B = 4; Sp = triton.cdiv(c["page_table"].shape[1], 128) * 128
Kc = torch.empty((B * 2 * Sp, 64), dtype=torch.uint8, device=dev); Vt = torch.empty((B * 2 * 64, Sp), dtype=torch.uint8, device=dev)
for nw in (2, 4, 8, 16):
    f = lambda: A._gather_kv_kernel[(Sp // 128, B)](c["k_cache"].view(torch.uint8), c["v_cache"].view(torch.uint8), c["page_table"], c["cache_seqlens"], Kc, Vt, c["page_table"].stride(0), Sp, PAGE=1, BLOCK=128, num_warps=nw)
    print(nw, f"{timeit(f):.1f} us")
