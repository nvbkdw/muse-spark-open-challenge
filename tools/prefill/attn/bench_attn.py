"""Correctness + timing of tools/prefill/attn/attn_impl.attention vs FA3 (sgl_kernel) at the prefill shapes.

  CUDA_VISIBLE_DEVICES=3 python tools/prefill/attn/bench_attn.py [--reps 30] [--only name,...] [--no-time]

Correctness: max abs / relative-L2 of [T,16,64] bf16 outputs, ours vs FA3, and (where cheap enough) both vs an FP32
SDPA reference on the same FP8 q/K/V. Timing: CUDA graph replay, cold L2 (512 MB buffer written before each replay),
median (same method as tools/prefill/bench_fa3.py). "ours" includes everything the call does (paged gather, kernel,
split combine). Data: reference-style scattered pages, "soft" keys (shared direction), queries partly aligned.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SHAPES, SCALE, dev, flops, make_case, timeit  # noqa: E402
import attn_impl as A  # noqa: E402
from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: E402

EXTRA = [  # correctness-only extras: kv_scale != 1 (descale path), variable lengths
    ("full_b2_3000_300_kvscale0.5", 2, 3000, 300, 0, 1, dict(kv_scale=0.5)),
    ("full_b3_5000_200_varlen", 3, 5000, 200, 0, 1, dict(var_lens=True)),
    ("full_b1_0_300", 1, 0, 300, 0, 1, {}),
]


def fa3(c):
    w = c["window"]
    descale = {}
    if c["kv_scale"] != 1.0:
        descale = dict(k_descale=c["k_scale"].reshape(1, 1).expand(c["B"], 2),
                       v_descale=c["v_scale"].reshape(1, 1).expand(c["B"], 2))
    return flash_attn_with_kvcache(q=c["q"], k_cache=c["k_cache"], v_cache=c["v_cache"], page_table=c["page_table"],
                                   cache_seqlens=c["cache_seqlens"], cu_seqlens_q=c["cu_seqlens_q"],
                                   max_seqlen_q=c["L"], softmax_scale=SCALE, causal=True,
                                   window_size=(w - 1, 0) if w else (-1, -1), **descale)


def ours_forced(c):  # the custom kernel regardless of the dispatch rule (full layers only)
    return A.custom_attention(c["q"], c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], c["L"], SCALE,
                              c["k_scale"].reshape(1).float(), c["v_scale"].reshape(1).float())


def ours(c):
    return A.attention(c["q"], c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], c["cu_seqlens_q"],
                       c["L"], SCALE, c["window"], c["k_scale"], c["v_scale"], kv_scale_is_one=c["kv_scale"] == 1.0)


def reference(c):
    B, L, page = c["B"], c["L"], c["page"]
    q = c["q"].float().view(B, L, 16, 64)
    kf = c["k_cache"].float().view(-1, 2, 64) * c["kv_scale"]
    vf = c["v_cache"].float().view(-1, 2, 64) * c["kv_scale"]
    outs = []
    for b in range(B):
        S = int(c["cache_seqlens"][b])
        seq = S - L
        p = torch.arange(S, device=dev)
        rows = c["page_table"][b].long()[p // page] * page + p % page
        k = kf[rows].repeat_interleave(8, 1).transpose(0, 1)
        v = vf[rows].repeat_interleave(8, 1).transpose(0, 1)
        qp = seq + torch.arange(L, device=dev)
        vis = p[None, :] <= qp[:, None]
        if c["window"]:
            vis &= p[None, :] > qp[:, None] - c["window"]
        o = torch.empty(L, 16, 64, device=dev)
        for j in range(0, L, 2048):
            sl = slice(j, min(j + 2048, L))
            o[sl] = torch.nn.functional.scaled_dot_product_attention(
                q[b, sl].transpose(0, 1), k, v, attn_mask=vis[sl][None], scale=SCALE).transpose(0, 1)
        outs.append(o)
    return torch.cat(outs)


def err(a, b):
    d = (a.float() - b.float())
    return d.abs().max().item(), (d.norm() / b.float().norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    A._compile()
    print(f"NVRTC compile + load: {time.time() - t0:.2f} s  (ptxas: "
          f"{[l.strip() for l in A._STATE['log'].splitlines() if 'Used' in l or 'C75' in l]})", flush=True)
    only = set(args.only.split(",")) if args.only else None
    cases = [(n, B, ctx, L, w, p, {}) for (n, B, ctx, L, w, p) in SHAPES] + EXTRA
    rows = []
    for name, B, ctx, L, w, p, kw in cases:
        if only and name not in only:
            continue
        c = make_case(B, ctx, L, w, p, **kw)
        path = "custom" if A.use_custom(w, L, B, c["page_table"].shape[1] * p) else "FA3 fallback"
        o = ours_forced(c) if w == 0 else ours(c)   # correctness: always the custom kernel on full layers
        f = fa3(c)
        torch.cuda.synchronize()
        nan = bool(torch.isnan(o).any() or torch.isinf(o).any())
        ma, rl = err(o, f)
        line = f"{name:28s} [{path:12s}] {'custom' if w == 0 else 'FA3':6s}-vs-FA3 maxabs {ma:.4f} relL2 {rl:.5f} nan/inf={nan}"
        work = B * L * (ctx + L)
        if not args.no_ref and work <= 2.2e8:
            r = reference(c)
            ma1, rl1 = err(o, r)
            ma2, rl2 = err(f, r)
            line += f" | vs fp32 ref: ours relL2 {rl1:.5f} (max {ma1:.4f}), FA3 relL2 {rl2:.5f} (max {ma2:.4f})"
            del r
        if not args.no_time and not kw:
            t_f = timeit(lambda: fa3(c), args.reps)
            t_o = timeit(lambda: ours(c), args.reps)
            fl = flops(c)
            line += (f" | FA3 {t_f:7.1f} us ({fl / t_f / 1e6:.0f} TF/s)  ours {t_o:7.1f} us ({fl / t_o / 1e6:.0f} TF/s)"
                     f"  speedup {t_f / t_o:.3f}")
            rows.append((name, path, t_f, t_o))
        print(line, flush=True)
        del c, o, f
        torch.cuda.empty_cache()
    if rows:
        print("\n| shape | path | FA3 us | ours us | speedup |\n|---|---|---|---|---|")
        for name, path, t_f, t_o in rows:
            print(f"| {name} | {path} | {t_f:.1f} | {t_o:.1f} | {t_f / t_o:.3f} |")


if __name__ == "__main__":
    main()
