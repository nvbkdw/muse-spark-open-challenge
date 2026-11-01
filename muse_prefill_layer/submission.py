# okc: problem=muse_prefill_layer
"""Muse Spark Open decoder layer, prefill shapes, one H200.

Pipeline (one CUDA graph per shape):
  qkv = x @ qkv_gate.T                      cuBLAS
  QK-norm + RoPE + FP8 q + FP8 K/V store    one Triton kernel
  attention                                 FA3 (sgl_kernel), FP8 q/K/V, paged
  per-head norm * sigmoid(gate)             Triton
  a = o_in @ o.T                            cuBLAS
  post-attn boundary + pre-FFN norm         Triton (writes stream1 fp32, h bf16)
  router hi/lo (one N=512 GEMM)             cuBLAS, side stream
  routing: sigmoid, biased top-8, renorm,   Triton (side stream); also assigns each local
     local-expert slot assignment                  route a row in its expert's slab (atomics)
  p = h @ pooler.T                          cuBLAS
  pre-expert norm + dispatch gather         Triton
  experts gate/up GEMM (NVFP4 x BF16)       humming grouped-masked wgmma kernel
  silu(g) * u                               Triton
  experts down GEMM (NVFP4 x BF16)          humming grouped-masked wgmma kernel
  post-expert norm + weighted combine       Triton
  f = moe @ unpooler.T                      cuBLAS
  post-FFN boundary + next pre-attn norm    Triton
"""
import json
import os

import torch
import triton
import triton.language as tl

os.environ.setdefault("HUMMING_CACHE_DIR", "/tmp/humming_cache_okc")
os.environ.setdefault("HUMMING_TMP_DIR", "/tmp/humming_tmp_okc")

from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: E402

HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
ROUTER_MULT = 0.1767766952966369
FP8 = torch.float8_e4m3fn
HQ, HKV, HD = 16, 2, 64


# ----------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------
@triton.jit
def _qk_rope_store_kernel(qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, kscale_ptr, vscale_ptr,
                          q_out_ptr, kc_ptr, vc_ptr, T,
                          HAS_ROPE: tl.constexpr, QKV_STRIDE: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < T
    i = tl.arange(0, 32)
    hq = tl.arange(0, 16)
    hk = tl.arange(0, 2)
    row = qkv_ptr + offs_t.to(tl.int64)[:, None, None] * QKV_STRIDE
    # q: [BT, 16, 32] even / odd
    q_off = hq[None, :, None] * 64 + 2 * i[None, None, :]
    m3 = tmask[:, None, None]
    qe = tl.load(row + q_off, mask=m3, other=0.0).to(tl.float32)
    qo = tl.load(row + q_off + 1, mask=m3, other=0.0).to(tl.float32)
    k_off = 1024 + hk[None, :, None] * 64 + 2 * i[None, None, :]
    ke = tl.load(row + k_off, mask=m3, other=0.0).to(tl.float32)
    ko = tl.load(row + k_off + 1, mask=m3, other=0.0).to(tl.float32)
    rq = tl.rsqrt((tl.sum(qe * qe, 2) + tl.sum(qo * qo, 2)) / 64.0 + 1e-5)
    rk = tl.rsqrt((tl.sum(ke * ke, 2) + tl.sum(ko * ko, 2)) / 64.0 + 1e-5)
    qe = qe * rq[:, :, None]
    qo = qo * rq[:, :, None]
    ke = ke * rk[:, :, None]
    ko = ko * rk[:, :, None]
    if HAS_ROPE:
        pos = tl.load(pos_ptr + offs_t, mask=tmask, other=0).to(tl.float32)
        inv = tl.load(inv_freq_ptr + i)
        ang = pos[:, None] * inv[None, :]
        c = tl.cos(ang)[:, None, :]
        s = tl.sin(ang)[:, None, :]
        qe, qo = qe * c - qo * s, qe * s + qo * c
        ke, ko = ke * c - ko * s, ke * s + ko * c
    qrow = q_out_ptr + offs_t.to(tl.int64)[:, None, None] * 1024 + q_off
    tl.store(qrow, qe.to(tl.float8e4nv), mask=m3)
    tl.store(qrow + 1, qo.to(tl.float8e4nv), mask=m3)
    loc = tl.load(loc_ptr + offs_t, mask=tmask, other=0)
    ks = tl.load(kscale_ptr)
    vs = tl.load(vscale_ptr)
    kv_off = hk[None, :, None] * 64 + 2 * i[None, None, :]
    krow = kc_ptr + loc[:, None, None] * 128 + kv_off
    tl.store(krow, (ke / ks).to(tl.float8e4nv), mask=m3)
    tl.store(krow + 1, (ko / ks).to(tl.float8e4nv), mask=m3)
    d = tl.arange(0, 128)
    v = tl.load(qkv_ptr + offs_t.to(tl.int64)[:, None] * QKV_STRIDE + 1152 + d[None, :], mask=tmask[:, None],
                other=0.0).to(tl.float32)
    tl.store(vc_ptr + loc[:, None] * 128 + d[None, :], (v / vs).to(tl.float8e4nv), mask=tmask[:, None])


@triton.jit
def _onorm_gate_kernel(o_ptr, qkv_ptr, out_ptr, T, QKV_STRIDE: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = offs_t < T
    h = tl.arange(0, 16)
    d = tl.arange(0, 64)
    off = h[None, :, None] * 64 + d[None, None, :]
    m3 = tmask[:, None, None]
    t64 = offs_t.to(tl.int64)[:, None, None]
    o = tl.load(o_ptr + t64 * 1024 + off, mask=m3, other=0.0).to(tl.float32)
    g = tl.load(qkv_ptr + t64 * QKV_STRIDE + 1280 + off, mask=m3, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(o * o, 2) / 64.0 + 1e-5)
    y = o * r[:, :, None] * tl.sigmoid(g)
    tl.store(out_ptr + t64 * 1024 + off, y.to(tl.bfloat16), mask=m3)


@triton.jit
def _attn_boundary_kernel(a_ptr, s_ptr, alpha_ptr, beta_ptr, w_ptr, s1_ptr, h_ptr, N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    a = tl.load(a_ptr + row * N + offs).to(tl.float32)
    s = tl.load(s_ptr + row * N + offs)
    ra = tl.rsqrt(tl.sum(a * a, 0) / N + 1e-8)
    s1 = tl.load(alpha_ptr + offs) * s + tl.load(beta_ptr + offs) * (a * ra)
    tl.store(s1_ptr + row * N + offs, s1)
    rs = tl.rsqrt(tl.sum(s1 * s1, 0) / N + 1e-5)
    h = s1 * rs * tl.load(w_ptr + offs)
    tl.store(h_ptr + row * N + offs, h.to(tl.bfloat16))


@triton.jit
def _ffn_boundary_kernel(f_ptr, s1_ptr, wf_ptr, alpha_ptr, beta_ptr, wn_ptr, s2_ptr, hid_ptr, N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    f = tl.load(f_ptr + row * N + offs).to(tl.float32) * tl.load(wf_ptr + offs)
    rf = tl.rsqrt(tl.sum(f * f, 0) / N + 1e-8)
    s1 = tl.load(s1_ptr + row * N + offs)
    s2 = tl.load(alpha_ptr + offs) * s1 + tl.load(beta_ptr + offs) * (f * rf)
    tl.store(s2_ptr + row * N + offs, s2)
    r2 = tl.rsqrt(tl.sum(s2 * s2, 0) / N + 1e-5)
    tl.store(hid_ptr + row * N + offs, (s2 * r2 * tl.load(wn_ptr + offs)).to(tl.bfloat16))


@triton.jit
def _routing_kernel(logit_ptr, bias_ptr, local_start_ptr, ids_ptr, w_ptr, rowmap_ptr, cnt_ptr, slab,
                    LOGIT_STRIDE: tl.constexpr, HILO: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, 256)
    x = tl.load(logit_ptr + t * LOGIT_STRIDE + e)
    if HILO:
        x += tl.load(logit_ptr + t * LOGIT_STRIDE + 256 + e)
    sc = tl.sigmoid(x * 0.1767766952966369)
    b = sc + tl.load(bias_ptr + e)
    j = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    ws = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        m = tl.max(b, 0)
        idx = tl.min(tl.where(b == m, e, 256), 0)
        wk = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(j == k, idx, ids)
        ws = tl.where(j == k, wk, ws)
        b = tl.where(e == idx, float("-inf"), b)
    ws = ws / (tl.sum(ws, 0) + 1e-15)
    tl.store(ids_ptr + t * 8 + j, ids)
    tl.store(w_ptr + t * 8 + j, ws)
    ls = tl.load(local_start_ptr)
    le = (ids - ls + 256) % 256
    is_local = le < 32
    slot = tl.atomic_add(cnt_ptr + le, tl.full([8], 1, tl.int32), mask=is_local)
    rm = tl.where(is_local, le * slab + slot, -1)
    tl.store(rowmap_ptr + t * 8 + j, rm)


@triton.jit
def _dispatch_kernel(p_ptr, w_ptr, rowmap_ptr, xg_ptr, N: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    p = tl.load(p_ptr + t * N + offs).to(tl.float32)
    r = tl.rsqrt(tl.sum(p * p, 0) / N + 1e-5)
    xn = (p * r * tl.load(w_ptr + offs)).to(tl.bfloat16)
    for j in tl.static_range(8):
        rm = tl.load(rowmap_ptr + t * 8 + j)
        if rm >= 0:
            tl.store(xg_ptr + rm.to(tl.int64) * N + offs, xn)


@triton.jit
def _silu_mul_kernel(h_ptr, rowmap_ptr, out_ptr, N: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    for j in tl.static_range(8):
        rm = tl.load(rowmap_ptr + t * 8 + j)
        if rm >= 0:
            r64 = rm.to(tl.int64)
            g = tl.load(h_ptr + r64 * (2 * N) + offs).to(tl.float32)
            u = tl.load(h_ptr + r64 * (2 * N) + N + offs).to(tl.float32)
            y = g * tl.sigmoid(g) * u
            tl.store(out_ptr + r64 * N + offs, y.to(tl.bfloat16))


@triton.jit
def _combine_kernel(y_ptr, rowmap_ptr, tw_ptr, w_ptr, out_ptr, N: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    wpe = tl.load(w_ptr + offs)
    acc = tl.zeros([N], dtype=tl.float32)
    for j in tl.static_range(8):
        rm = tl.load(rowmap_ptr + t * 8 + j)
        if rm >= 0:
            y = tl.load(y_ptr + rm.to(tl.int64) * N + offs).to(tl.float32) * wpe
            r = tl.rsqrt(tl.sum(y * y, 0) / N + 1e-8)
            acc += (tl.load(tw_ptr + t * 8 + j) * r) * y
    tl.store(out_ptr + t * N + offs, acc.to(tl.bfloat16))


# ----------------------------------------------------------------------------------------
# humming NVFP4 grouped GEMM
# ----------------------------------------------------------------------------------------
def _humming_prepare(w, sf, gs):
    """Repack one NVFP4 expert bank (E, N, K/2) into humming's layout (lossless)."""
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema
    from humming.schema.humming import HummingInputSchema
    from humming.transform import prepare_layer_config, transform_humming_tensors
    from humming import dtypes
    e, n, k2 = w.shape
    k = 2 * k2
    schema = ModeloptNvfp4WeightSchema()
    wschema, tensors = schema.convert_humming(
        tensors={"weight": w.contiguous(), "weight_scale": sf.contiguous(), "weight_scale_2": gs.contiguous()},
        shape_n_stacks=[n], shape_k_stacks=[k], param_dtype=torch.bfloat16, num_experts=e)
    ischema = HummingInputSchema(a_dtype=dtypes.bfloat16)
    cfg = prepare_layer_config(shape_n=n, shape_k=k, weight_schema=wschema, input_schema=ischema, num_experts=e,
                               pad_n_to_multiple=1, pad_k_to_multiple=1, torch_dtype=torch.bfloat16, has_bias=False)
    tensors = transform_humming_tensors(cfg, tensors)
    return {"config": cfg.to_str(), "tensors": tensors, "n": n, "k": k}


_CC = json.dumps({"gemm_type": "grouped_masked"})


def _gemm_cfg(bm, bn=256, bk=64, stages=4, stream_k=True):
    return {"block_shape": (bm, bn, bk), "warp_shape": (bm, 32, 64), "use_stream_k": stream_k,
            "use_f16_accum": False, "num_stages": stages, "use_warp_spec": True, "use_tma": True,
            "use_mbarrier": True}


def _tuning_for(tokens, which):
    """Per-shape kernel config (by tokens in the batch)."""
    if tokens <= 1024:
        return _gemm_cfg(48, 256, 128)
    if tokens <= 4096:
        return _gemm_cfg(128, 256, 64)
    return _gemm_cfg(176, 256, 64)


def _humming_run(bank, x, out, counts, tuning, valid_m):
    from humming import ops
    t = bank["tensors"]
    return ops.humming_gemm(
        layer_config=bank["config"], compute_config=_CC, tuning_config=json.dumps(tuning),
        inputs=x, weight=t["weight"], weight_scale=t["weight_scale"], weight_scale_2=t.get("weight_scale_2"),
        outputs=out, expert_layout=counts, locks=bank["locks"], top_k=1, valid_shape_m=valid_m)


# ----------------------------------------------------------------------------------------
def _coeffs(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha.contiguous(), beta.contiguous()


def prepare(weights):
    dev = weights["qkv_gate"].device
    st = {}
    st["qkv_gate"] = weights["qkv_gate"].contiguous()
    st["o"] = weights["o"].contiguous()
    st["pooler"] = weights["pooler"].contiguous()
    st["unpooler"] = weights["unpooler"].contiguous()
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    st["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [8192, 512]
    st["router_bias"] = weights["router_bias"].float().contiguous()
    st["a1"], st["b1"] = _coeffs(weights["attn_residual_gate"])
    st["a2"], st["b2"] = _coeffs(weights["ffn_residual_gate"])
    st["w_pre_ffn"] = (1.0 + weights["pre_ffn_norm"].float()).contiguous()
    st["w_pre_exp"] = (1.0 + weights["pre_expert_norm"].float()).contiguous()
    st["w_post_exp"] = (1.0 + weights["post_expert_norm"].float()).contiguous()
    st["w_post_ffn"] = (1.0 + weights["post_ffn_norm"].float()).contiguous()
    st["w_next"] = (1.0 + weights["next_pre_attn_norm"].float()).contiguous()
    st["inv_freq"] = (1.0 / (500000.0 ** (torch.arange(0, 64, 2, dtype=torch.float, device=dev) / 64))).contiguous()
    st["w13"] = _humming_prepare(weights["w13"], weights["w13_sf"], weights["w13_gs"])
    st["w2"] = _humming_prepare(weights["w2"], weights["w2_sf"], weights["w2_gs"])
    st["w13"]["locks"] = torch.zeros(1024, dtype=torch.int32, device=dev)
    st["w2"]["locks"] = torch.zeros(1024, dtype=torch.int32, device=dev)
    st["cnt"] = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    st["side"] = torch.cuda.Stream(dev)
    _buffers(st, 16384, dev)
    # Pre-build (JIT) the expert GEMM kernels for the token counts we expect.
    from humming.kernel.humming import HummingKernel
    for tokens in (300, 753, 4096, 16384):
        for name in ("w13", "w2"):
            HummingKernel.prepare_kernels(st[name]["config"], _CC, json.dumps(_tuning_for(tokens, name)))
    torch.cuda.synchronize()
    return st


def _buffers(st, tokens, dev):
    """Scratch views for this token count, carved from one buffer set sized for the largest
    count seen (allocated in prepare for 16384 tokens; grown if a larger case shows up)."""
    cap = st.get("cap", 0)
    if tokens > cap:
        rows = LOCAL_EXPERTS * tokens
        st["xg_buf"] = torch.empty(rows * D_EXPERT, dtype=torch.bfloat16, device=dev)
        st["h13_buf"] = torch.empty(rows * 2 * D_EXPERT, dtype=torch.bfloat16, device=dev)
        st["inter_buf"] = torch.empty(rows * D_EXPERT, dtype=torch.bfloat16, device=dev)
        st["cap"] = tokens
    slab = tokens  # an expert gets at most one route per token
    rows = LOCAL_EXPERTS * slab
    xg = st["xg_buf"][: rows * D_EXPERT].view(rows, D_EXPERT)
    return {"slab": slab, "xg": xg, "y": xg,  # xg is dead after the gate/up GEMM
            "h13": st["h13_buf"][: rows * 2 * D_EXPERT].view(rows, 2 * D_EXPERT),
            "inter": st["inter_buf"][: rows * D_EXPERT].view(rows, D_EXPERT)}


def kernel_fn(data, st):
    cfg = data["config"]
    T = cfg["tokens"]
    batch, length = cfg["batch"], cfg["q_len"]
    x = data["attn_input"]
    dev = x.device
    bufs = _buffers(st, T, dev)

    # ---- attention ----
    qkv = torch.mm(x, st["qkv_gate"].t())  # [T, 2304] bf16
    q8 = torch.empty((T, HQ, HD), dtype=FP8, device=dev)
    kc, vc = data["k_cache"], data["v_cache"]
    BT = 4
    _qk_rope_store_kernel[(triton.cdiv(T, BT),)](
        qkv, data["positions"], data["cache_loc"], st["inv_freq"], data["k_scale"], data["v_scale"],
        q8, kc, vc, T,
        HAS_ROPE=bool(cfg["rope_theta"]), QKV_STRIDE=2304, BLOCK_T=BT, num_warps=4)
    window = cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, HKV),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, HKV))
    out = flash_attn_with_kvcache(
        q=q8, k_cache=kc, v_cache=vc, page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"],
        **descale)
    if isinstance(out, tuple):
        out = out[0]
    o_in = torch.empty((T, 1024), dtype=torch.bfloat16, device=dev)
    _onorm_gate_kernel[(triton.cdiv(T, BT),)](out, qkv, o_in, T, QKV_STRIDE=2304, BLOCK_T=BT, num_warps=4)
    a = torch.mm(o_in, st["o"].t())  # [T, 8192] bf16

    # ---- post-attention boundary ----
    s1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    h = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _attn_boundary_kernel[(T,)](a, data["stream"], st["a1"], st["b1"], st["w_pre_ffn"], s1, h, N=HIDDEN,
                                num_warps=16)

    # ---- router (side stream) || pooler ----
    main, side = torch.cuda.current_stream(), st["side"]
    cnt = st["cnt"]
    cnt.zero_()
    side.wait_stream(main)
    ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    tw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
    rowmap = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    with torch.cuda.stream(side):
        logits = torch.mm(h, st["router_hilo_t"], out_dtype=torch.float32)  # [T, 512]
        _routing_kernel[(T,)](logits, st["router_bias"], data["local_start"], ids, tw, rowmap, cnt, bufs["slab"],
                              LOGIT_STRIDE=512, HILO=True, num_warps=2)
    p = torch.mm(h, st["pooler"].t())  # [T, 4096]
    main.wait_stream(side)

    # ---- experts ----
    slab = bufs["slab"]
    rows = LOCAL_EXPERTS * slab
    xg, h13, inter, y = bufs["xg"], bufs["h13"], bufs["inter"], bufs["y"]
    _dispatch_kernel[(T,)](p, st["w_pre_exp"], rowmap, xg, N=D_EXPERT, num_warps=8)
    valid_m = max(T * 2, 1)
    _humming_run(st["w13"], xg, h13, cnt, _tuning_for(T, "w13"), valid_m)
    _silu_mul_kernel[(T,)](h13, rowmap, inter, N=D_EXPERT, num_warps=8)
    _humming_run(st["w2"], inter, y, cnt, _tuning_for(T, "w2"), valid_m)
    moe = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    _combine_kernel[(T,)](y, rowmap, tw, st["w_post_exp"], moe, N=D_EXPERT, num_warps=8)

    # ---- unpooler + FFN boundary ----
    f = torch.mm(moe, st["unpooler"].t())  # [T, 8192]
    s2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hid = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _ffn_boundary_kernel[(T,)](f, s1, st["w_post_ffn"], st["a2"], st["b2"], st["w_next"], s2, hid, N=HIDDEN,
                               num_warps=16)
    return hid, s2, ids
