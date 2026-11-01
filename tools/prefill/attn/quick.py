import sys, torch
from common import *
import attn_impl as A
from sgl_kernel.flash_attn import flash_attn_with_kvcache
def fa3(c):
    w = c["window"]
    return flash_attn_with_kvcache(q=c["q"], k_cache=c["k_cache"], v_cache=c["v_cache"], page_table=c["page_table"],
        cache_seqlens=c["cache_seqlens"], cu_seqlens_q=c["cu_seqlens_q"], max_seqlen_q=c["max_seqlen_q"],
        softmax_scale=SCALE, causal=True, window_size=(w - 1, 0) if w else (-1, -1))
def ref(c):
    # fp32 reference with FP8 q,k,v
    B, L = c["B"], c["L"]; page = c["page"]
    q = c["q"].float().view(B, L, 16, 64)
    kf = c["k_cache"].float().view(-1, 2, 64); vf = c["v_cache"].float().view(-1, 2, 64)
    outs = []
    for b in range(B):
        S = int(c["cache_seqlens"][b]); seq = S - L
        p = torch.arange(S, device=dev)
        rows = c["page_table"][b].long()[p // page] * page + p % page
        k = kf[rows].repeat_interleave(8, 1).transpose(0, 1); v = vf[rows].repeat_interleave(8, 1).transpose(0, 1)
        qp = seq + torch.arange(L, device=dev)
        vis = p[None, :] <= qp[:, None]
        if c["window"]: vis &= p[None, :] > qp[:, None] - c["window"]
        outs.append(torch.nn.functional.scaled_dot_product_attention(q[b].transpose(0, 1), k, v, attn_mask=vis[None], scale=SCALE).transpose(0, 1))
    return torch.cat(outs).view(B * L, 16, 64)
args = [int(x) for x in sys.argv[1:5]]; ns = int(sys.argv[5]) if len(sys.argv) > 5 else None
B, ctx, L, page = args
c = make_case(B, ctx, L, 0, page)
o = A.custom_attention(c["q"], c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], L, SCALE, c["k_scale"], c["v_scale"], num_splits=ns)
f = fa3(c); torch.cuda.synchronize()
r = ref(c) if B * L * (ctx + L) < 3e8 else None
def err(a, b): return f"maxabs {(a.float()-b.float()).abs().max().item():.4f} rel_l2 {((a.float()-b.float()).norm()/b.float().norm()).item():.5f}"
print("mine vs fa3:", err(o, f))
if r is not None: print("mine vs ref:", err(o, r), "| fa3 vs ref:", err(f, r))
print("nan:", torch.isnan(o).any().item())
