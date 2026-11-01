import torch
from common import *
from sgl_kernel.flash_attn import flash_attn_with_kvcache
c = make_case(1, 32768, 16384, 0, 1)
f = lambda: flash_attn_with_kvcache(q=c["q"], k_cache=c["k_cache"], v_cache=c["v_cache"], page_table=c["page_table"],
    cache_seqlens=c["cache_seqlens"], cu_seqlens_q=c["cu_seqlens_q"], max_seqlen_q=c["max_seqlen_q"], softmax_scale=SCALE, causal=True)
f(); torch.cuda.synchronize(); f(); torch.cuda.synchronize()
