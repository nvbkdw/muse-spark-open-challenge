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


# ----------------------------------------------------------------------------------------
# Attention: custom FP8 GQA-packed wgmma kernel (NVRTC, driver-API launch) for full layers,
# FA3 for sliding-window layers. Source: tools/prefill/attn/attn_impl.py
# ----------------------------------------------------------------------------------------
import ctypes  # noqa: E402
import math  # noqa: E402
BN = 128          # keys per tile
QPB = 24          # queries per CTA (x 8 heads = 192 rows = 3 consumer warpgroups x 64)
STAGES = 10
CFG = dict(ROWSUM=2, MAXTREE=1, POLY=0, EXPFN='ex2', F16SM=0)
NTHREADS = 512    # 3 consumer warpgroups + 1 producer warpgroup
LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------------------------------------------
# CUDA source
# ----------------------------------------------------------------------------------------------------------------
def _wgmma_ss(n):
    regs = ", ".join(f"%{i}" for i in range(n // 2))
    na = n // 2
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 2}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 {{{regs}}}, %{na}, %{na + 1}, p, 1, 1;\\n}}\\n"'
            f' : ' + ", ".join(f'"+f"(d[{i}])' for i in range(n // 2)) +
            ' : "l"(da), "l"(db), "r"(scale_d));')


def _wgmma_rs(n):
    na = n // 2
    regs = ", ".join(f"%{i}" for i in range(na))
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{na + 5}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 {{{regs}}}, {{%{na}, %{na + 1}, %{na + 2}, %{na + 3}}}, %{na + 4}, p, 1, 1;\\n}}\\n"'
            f' : ' + ", ".join(f'"+f"(d[{i}])' for i in range(na)) +
            ' : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(scale_d));')


_CUDA_SRC = r"""
typedef unsigned long long u64; typedef unsigned int u32; typedef unsigned short u16; typedef unsigned char u8;
struct __align__(64) TmaDesc { u64 x[16]; };

#define BN 128
#define QPB 24
#define STAGES __STAGES__
#define ROWSUM __ROWSUM__
#define MAXTREE __MAXTREE__
#define POLY __POLY__
#define F16SM __F16SM__
#define TPI 1
#define EXP_FN __EXPFN__
#define VROWS 64
#define KBYTES (BN * 64)
#define VBYTES (VROWS * BN)
#define QBYTES (QPB * 8 * 64)

__device__ __forceinline__ u32 smem_u32(const void* p) {
  u32 r; asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(r) : "l"(p)); return r;
}
__device__ __forceinline__ void mbar_init(u32 bar, u32 cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(u32 bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ u32 mbar_try(u32 bar, u32 ph) {
  u32 ok; asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                       : "=r"(ok) : "r"(bar), "r"(ph) : "memory"); return ok; }
__device__ __forceinline__ void mbar_wait(u32 bar, u32 ph) { while (!mbar_try(bar, ph)) {} }
__device__ __forceinline__ void tma2d(u32 dst, const TmaDesc* d, u32 bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(d), "r"(bar), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void tma4d(u32 dst, const TmaDesc* d, u32 bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
               :: "r"(dst), "l"(d), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_reg(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ u32 pack4(float x0, float x1, float x2, float x3) {
  u16 lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return (u32)lo | ((u32)hi << 16);
}
__device__ __forceinline__ u32 f2h2(float lo, float hi) {
  u32 r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ u32 hmax2(u32 a, u32 b) { u32 r; asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ u32 hfma2(u32 a, u32 b, u32 c) { u32 r; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ u32 hex2(u32 a) { u32 r; asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(a)); return r; }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r; }
__device__ __forceinline__ float h_lo(u32 a) { float r; asm("{ .reg .f16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, l; }" : "=f"(r) : "r"(a)); return r; }
__device__ __forceinline__ float h_hi(u32 a) { float r; asm("{ .reg .f16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, h; }" : "=f"(r) : "r"(a)); return r; }
__device__ __forceinline__ u32 h2e4(u32 lo, u32 hi) {   // two f16x2 -> four e4m3 bytes (lo pair in the low half)
  u16 a, b;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(a) : "r"(lo));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(b) : "r"(hi));
  return (u32)a | ((u32)b << 16); }
__device__ __forceinline__ u32 bf2(float a, float b) {  // a -> low half
  u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a)); return r; }

// smem matrix descriptors (K-major, swizzled)
__device__ __forceinline__ u64 desc64(u32 addr) {   // 64B swizzle, 64-byte rows, 8-row groups 512 B apart
  return (u64)((addr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(512 >> 4) << 32) | ((u64)2 << 62); }
__device__ __forceinline__ u64 desc128(u32 addr) {  // 128B swizzle, 128-byte rows, 8-row groups 1024 B apart
  return (u64)((addr & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)(1024 >> 4) << 32) | ((u64)1 << 62); }

__device__ __forceinline__ void mma_s(float* d, u64 da, u64 db, int scale_d) {
  __WGMMA_SS_64__
}
__device__ __forceinline__ void mma_s128(float* d, u64 da, u64 db, int scale_d) {
  __WGMMA_SS_128__
}
__device__ __forceinline__ void mma_pv(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u64 db, int scale_d) {
  __WGMMA_RS_64__
}
__device__ __forceinline__ void mma_rs(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u64 db, int scale_d) {
  __WGMMA_RS_8__
}
__device__ __forceinline__ float ex2_poly(float x) {   // 2^x on the FMA pipe (x <= ~8); rel err ~2e-4
  x = fmaxf(x, -127.f);
  const float t = x + 12582912.f;
  const float j = t - 12582912.f;
  const float f = x - j;
  const float p = fmaf(fmaf(fmaf(0.054848002280439f, f, 0.24180660514564356f), f, 0.6932481954581833f), f, 0.9999886556839822f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}

extern "C" __global__ void __launch_bounds__(512, 1)
attn_fwd(const __grid_constant__ TmaDesc tmK, const __grid_constant__ TmaDesc tmV, const __grid_constant__ TmaDesc tmQ,
         const int* __restrict__ cache_seqlens, u16* __restrict__ out, float* __restrict__ opart, float* __restrict__ mlpart,
         const float* __restrict__ kscale, const float* __restrict__ vscale,
         int L, int T, int Spad, int nsplit, float sl2_base)
{
  extern __shared__ __align__(1024) u8 smem_raw[];
  u8* smem = (u8*)(((unsigned long long)smem_raw + 1023) & ~1023ull);
  u8* Vs = smem;                                  // STAGES x 9216
  u8* Ks = Vs + STAGES * VBYTES;                  // STAGES x 8192
  u8* Qs = Ks + STAGES * KBYTES;                  // 12288
  u8* Os = Qs + QBYTES;                           // 1024: 8 x 128 B K-major "ones" operand (row 0 = 1.0)
  u64* bars = (u64*)(Os + 1024);                  // full[STAGES], empty[STAGES], qbar

  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int b = blockIdx.z;
  const int g = blockIdx.y & 1;
  const int split = blockIdx.y >> 1;
  const int mb = gridDim.x - 1 - blockIdx.x;      // heaviest query blocks first
  const int cs = cache_seqlens[b];
  const int seq = cs - L;
  const int q_first = mb * QPB;
  const int q_last = min(q_first + QPB - 1, L - 1);
  const int kv_end = seq + q_last + 1;
  const int ntiles_all = (kv_end + BN - 1) / BN;
  const int tps = (ntiles_all + nsplit - 1) / nsplit;
  const int t_begin = split * tps;
  const int t_end = min(t_begin + tps, ntiles_all);
  const int ntiles = t_end - t_begin;
  const int ntiles_eff = (ntiles + TPI - 1) / TPI * TPI;   // padded tiles are fully masked

  if (ntiles <= 0) {   // empty split: m = -inf, l = 0
    if (wg < 3) {
      int r = wg * 64 + (tid & 127) / 2;
      int qi = q_first + r / 8;
      if ((tid & 1) == 0 && qi < L) {
        size_t row = ((size_t)split * T + (size_t)b * L + qi) * 16 + 8 * g + (r & 7);
        mlpart[row * 2] = __int_as_float(0xff800000);
        mlpart[row * 2 + 1] = 0.f;
        for (int c = 0; c < 64; c++) opart[row * 64 + c] = 0.f;
      }
    }
    return;
  }

  const u32 full0 = smem_u32(bars), empty0 = smem_u32(bars + STAGES), qbar = smem_u32(bars + 2 * STAGES);
  if (tid == 0) {
    for (int s = 0; s < STAGES; s++) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, 12); }
    mbar_init(qbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // constant "ones" row (d = 64) and zero rows 65..71 of every V^T stage: the PV wgmma then yields the row sum
  for (int i = tid; i < 256; i += blockDim.x) ((u32*)Os)[i] = (i < 32) ? 0x38383838u : 0u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  if (wg == 3) {
    // ---------------- producer ----------------
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    if (tid == 384) {
      mbar_expect_tx(qbar, QBYTES);
      tma4d(smem_u32(Qs), &tmQ, qbar, 0, 0, g, b * L + q_first);
      const int krow0 = (b * 2 + g) * Spad, vrow0 = (b * 2 + g) * 64;
      for (int it = 0; it < ntiles_eff; it++) {
        int s = it % STAGES;
        if (it >= STAGES) mbar_wait(empty0 + 8 * s, ((it / STAGES) - 1) & 1);
        u32 fb = full0 + 8 * s;
        mbar_expect_tx(fb, KBYTES + 64 * BN);
        int kt = (t_begin + min(it, ntiles - 1)) * BN;
        tma2d(smem_u32(Ks + s * KBYTES), &tmK, fb, 0, krow0 + kt);
        tma2d(smem_u32(Vs + s * VBYTES), &tmV, fb, kt, vrow0);
      }
    }
    return;
  }

  // ---------------- consumers ----------------
  asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
  const int t = tid & 127, w = t >> 5, lane = t & 31, quad = lane & 3;
  const int r0 = wg * 64 + w * 16 + (lane >> 2);          // row of d[0..1] (the other row is r0 + 8)
  const int qi0 = q_first + r0 / 8, qi1 = qi0 + 1;
  const int lim0 = seq + min(qi0, L - 1) + 1, lim1 = seq + min(qi1, L - 1) + 1;  // exclusive key limits
  const int lim_min = seq + q_first + 1;
  const float sl2 = sl2_base * kscale[0];
  const float NEG_INF = __int_as_float(0xff800000);

  float o[32];
#pragma unroll
  for (int i = 0; i < 32; i++) o[i] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF;
  float l0 = 0.f, l1 = 0.f;
  float rs[4] = {0.f, 0.f, 0.f, 0.f};
  const u64 dones = desc128(smem_u32(Os));
  float s[64];
  u32 pa[16];
  const u64 dq = desc64(smem_u32(Qs + wg * 64 * 64));
  float al0 = 1.f, al1 = 1.f;   // pending O rescale (applied right before the next PV)
  const u32 sl2h = f2h2(sl2, sl2);
  const float isl2 = 1.f / sl2;
  mbar_wait(qbar, 0);

  // softmax on s (tile `tt`): mask, running max, exponentials (P scaled by 2^8); O untouched
#if F16SM
  // f16 softmax: scores -> f16x2 pairs (same row), packed max, HFMA2 for the exponent argument, ex2.f16x2,
  // e4m3x2.f16x2. The subtracted max is snapped so that the f16 bias is exact (m_eff), keeping tiles consistent.
  u32 hp[32];
#define SOFTMAX_PART1(tt)                                                                                    \
  {                                                                                                          \
    const int kbase = (t_begin + (tt)) * BN;                                                                 \
    if (kbase + BN > lim_min) {                                                                              \
      _Pragma("unroll") for (int i = 0; i < 64; i++) {                                                       \
        int key = kbase + 8 * (i >> 2) + 2 * quad + (i & 1);                                                 \
        if (key >= ((i & 2) ? lim1 : lim0)) s[i] = NEG_INF;                                                  \
      }                                                                                                      \
    }                                                                                                        \
    _Pragma("unroll") for (int k = 0; k < 32; k++) hp[k] = f2h2(s[2 * k], s[2 * k + 1]);                     \
    /* hp[k]: row0 if (k & 1) == 0, row1 otherwise */                                                        \
    u32 ra = hmax2(hp[0], hp[2]), rb = hmax2(hp[1], hp[3]), rc = hmax2(hp[4], hp[6]), rd = hmax2(hp[5], hp[7]); \
    _Pragma("unroll") for (int k = 8; k < 32; k += 4) {                                                      \
      ra = hmax2(ra, hp[k]); rb = hmax2(rb, hp[k + 1]); rc = hmax2(rc, hp[k + 2]); rd = hmax2(rd, hp[k + 3]); \
    }                                                                                                        \
    ra = hmax2(ra, rc); rb = hmax2(rb, rd);                                                                  \
    /* fold halves: (r0max, r1max) */                                                                        \
    u32 mm = hmax2(prmt(ra, rb, 0x5410), prmt(ra, rb, 0x7632));                                              \
    mm = hmax2(mm, __shfl_xor_sync(0xffffffff, mm, 1));                                                     \
    mm = hmax2(mm, __shfl_xor_sync(0xffffffff, mm, 2));                                                     \
    const float mx0 = h_lo(mm), mx1 = h_hi(mm);                                                              \
    float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);   /* raw score units */                                \
    u32 bb;                                                                                                  \
    {                                                                                                        \
      const float b0 = (mn0 == NEG_INF) ? 0.f : 8.f - mn0 * sl2;                                             \
      const float b1 = (mn1 == NEG_INF) ? 0.f : 8.f - mn1 * sl2;                                             \
      bb = f2h2(b0, b1);                                                                                     \
      if (mn0 != NEG_INF) mn0 = (8.f - h_lo(bb)) * isl2;                                                     \
      if (mn1 != NEG_INF) mn1 = (8.f - h_hi(bb)) * isl2;                                                     \
    }                                                                                                        \
    al0 = (mn0 == NEG_INF || mn0 == m0) ? 1.f : ex2((m0 - mn0) * sl2);                                       \
    al1 = (mn1 == NEG_INF || mn1 == m1) ? 1.f : ex2((m1 - mn1) * sl2);                                       \
    m0 = mn0; m1 = mn1;                                                                                      \
    const u32 b00 = prmt(bb, bb, 0x1010), b11 = prmt(bb, bb, 0x3232);                                        \
    _Pragma("unroll") for (int k = 0; k < 32; k++) hp[k] = hex2(hfma2(hp[k], sl2h, (k & 1) ? b11 : b00));    \
  }
#define PACK_P()                                                                                             \
  _Pragma("unroll") for (int c = 0; c < 4; c++) {                                                            \
    const u32* x = hp + 8 * c;                                                                               \
    pa[4 * c + 0] = h2e4(x[0], x[2]);                                                                        \
    pa[4 * c + 1] = h2e4(x[1], x[3]);                                                                        \
    pa[4 * c + 2] = h2e4(x[4], x[6]);                                                                        \
    pa[4 * c + 3] = h2e4(x[5], x[7]);                                                                        \
  }
#else
#define SOFTMAX_PART1(tt)                                                                                    \
  {                                                                                                          \
    const int kbase = (t_begin + (tt)) * BN;                                                                 \
    if (kbase + BN > lim_min) {                                                                              \
      _Pragma("unroll") for (int i = 0; i < 64; i++) {                                                       \
        int key = kbase + 8 * (i >> 2) + 2 * quad + (i & 1);                                                 \
        if (key >= ((i & 2) ? lim1 : lim0)) s[i] = NEG_INF;                                                  \
      }                                                                                                      \
    }                                                                                                        \
    float xa = fmaxf(s[0], s[4]), xb = fmaxf(s[1], s[5]), ya = fmaxf(s[2], s[6]), yb = fmaxf(s[3], s[7]);   \
    _Pragma("unroll") for (int i = 8; i < 64; i += 8) {                                                      \
      xa = fmaxf(xa, fmaxf(s[i], s[i + 4])); xb = fmaxf(xb, fmaxf(s[i + 1], s[i + 5]));                      \
      ya = fmaxf(ya, fmaxf(s[i + 2], s[i + 6])); yb = fmaxf(yb, fmaxf(s[i + 3], s[i + 7]));                  \
    }                                                                                                        \
    float mx0 = fmaxf(xa, xb), mx1 = fmaxf(ya, yb);                                                          \
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));                                                   \
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));                                                   \
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));                                                   \
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));                                                   \
    const float mn0 = fmaxf(m0, mx0 * sl2), mn1 = fmaxf(m1, mx1 * sl2);                                      \
    al0 = (mn0 == NEG_INF || mn0 == m0) ? 1.f : ex2(m0 - mn0);                                               \
    al1 = (mn1 == NEG_INF || mn1 == m1) ? 1.f : ex2(m1 - mn1);                                               \
    m0 = mn0; m1 = mn1;                                                                                      \
    const float nb0 = (m0 == NEG_INF) ? 0.f : 8.f - m0;                                                      \
    const float nb1 = (m1 == NEG_INF) ? 0.f : 8.f - m1;                                                      \
    _Pragma("unroll") for (int i = 0; i < 64; i++) {                                                         \
      const float x = fmaf(s[i], sl2, (i & 2) ? nb1 : nb0);                                                  \
      s[i] = ((i & 15) < POLY) ? ex2_poly(x) : EXP_FN(x);                                                    \
    }                                                                                                        \
  }
#define PACK_P()                                                                                             \
  _Pragma("unroll") for (int c = 0; c < 4; c++) {                                                            \
    const float* x = s + 16 * c;                                                                             \
    pa[4 * c + 0] = pack4(x[0], x[1], x[4], x[5]);                                                           \
    pa[4 * c + 1] = pack4(x[2], x[3], x[6], x[7]);                                                           \
    pa[4 * c + 2] = pack4(x[8], x[9], x[12], x[13]);                                                         \
    pa[4 * c + 3] = pack4(x[10], x[11], x[14], x[15]);                                                       \
  }
#endif
#define ISSUE_PV(stage)                                                                                      \
  {                                                                                                          \
    if (__any_sync(0xffffffff, (al0 != 1.f) || (al1 != 1.f))) {                                              \
      _Pragma("unroll") for (int i = 0; i < 32; i += 4) { o[i] *= al0; o[i + 1] *= al0; o[i + 2] *= al1; o[i + 3] *= al1; } \
    }                                                                                                        \
    const u64 dv = desc128(smem_u32(Vs + (stage) * VBYTES));                                                 \
    wg_fence();                                                                                              \
    _Pragma("unroll") for (int c = 0; c < 4; c++) mma_pv(o, pa[4 * c], pa[4 * c + 1], pa[4 * c + 2], pa[4 * c + 3], dv + 2 * c, 1); \
    _Pragma("unroll") for (int c = 0; c < 4; c++) mma_rs(rs, pa[4 * c], pa[4 * c + 1], pa[4 * c + 2], pa[4 * c + 3], dones + 2 * c, c > 0); \
    wg_commit();                                                                                             \
  }

  // tile 0
  mbar_wait(full0, 0);
  {
    const u64 dk = desc64(smem_u32(Ks));
    wg_fence();
    mma_s128(s, dq, dk, 0); mma_s128(s, dq + 2, dk + 2, 1);
    wg_commit();
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < 64; i++) fence_reg(s[i]);
  }
  SOFTMAX_PART1(0)
  PACK_P()
  al0 = 1.f; al1 = 1.f;   // O is zero: nothing to rescale

  for (int it = 1; it < ntiles; it++) {
    const int st = it % STAGES, sp = (it - 1) % STAGES;
    mbar_wait(full0 + 8 * st, (it / STAGES) & 1);
    {
      const u64 dk = desc64(smem_u32(Ks + st * KBYTES));
      wg_fence();
      mma_s128(s, dq, dk, 0); mma_s128(s, dq + 2, dk + 2, 1);
      wg_commit();
    }
    ISSUE_PV(sp)
    wg_wait<1>();
#pragma unroll
    for (int i = 0; i < 64; i++) fence_reg(s[i]);
    const float pm0 = m0, pm1 = m1;
    SOFTMAX_PART1(it)
    (void)pm0; (void)pm1;
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; i++) fence_reg(o[i]);
#pragma unroll
    for (int i = 0; i < 4; i++) fence_reg(rs[i]);
    if (lane == 0) mbar_arrive(empty0 + 8 * sp);
    l0 = (l0 + rs[0]) * al0; l1 = (l1 + rs[2]) * al1;
    PACK_P()
  }
  ISSUE_PV((ntiles - 1) % STAGES)
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < 32; i++) fence_reg(o[i]);
#pragma unroll
  for (int i = 0; i < 4; i++) fence_reg(rs[i]);
  l0 += rs[0]; l1 += rs[2];
  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);

  const int h0 = 8 * g + (r0 & 7);
#if F16SM
  if (m0 != NEG_INF) m0 *= sl2;   // split partials carry the max in log2 units
  if (m1 != NEG_INF) m1 *= sl2;
#endif
  if (nsplit == 1) {
    const float vs = vscale[0];
    const float i0 = (l0 > 0.f) ? vs / l0 : 0.f, i1 = (l1 > 0.f) ? vs / l1 : 0.f;
    if (qi0 < L) {
      u32* p = (u32*)(out + (((size_t)b * L + qi0) * 16 + h0) * 64);
#pragma unroll
      for (int j = 0; j < 8; j++) p[4 * j + quad] = bf2(o[4 * j] * i0, o[4 * j + 1] * i0);
    }
    if (qi1 < L) {
      u32* p = (u32*)(out + (((size_t)b * L + qi1) * 16 + h0) * 64);
#pragma unroll
      for (int j = 0; j < 8; j++) p[4 * j + quad] = bf2(o[4 * j + 2] * i1, o[4 * j + 3] * i1);
    }
  } else {
    if (qi0 < L) {
      size_t row = ((size_t)split * T + (size_t)b * L + qi0) * 16 + h0;
      float2* p = (float2*)(opart + row * 64);
#pragma unroll
      for (int j = 0; j < 8; j++) p[4 * j + quad] = make_float2(o[4 * j], o[4 * j + 1]);
      if (quad == 0) { mlpart[row * 2] = m0; mlpart[row * 2 + 1] = l0; }
    }
    if (qi1 < L) {
      size_t row = ((size_t)split * T + (size_t)b * L + qi1) * 16 + h0;
      float2* p = (float2*)(opart + row * 64);
#pragma unroll
      for (int j = 0; j < 8; j++) p[4 * j + quad] = make_float2(o[4 * j + 2], o[4 * j + 3]);
      if (quad == 0) { mlpart[row * 2] = m1; mlpart[row * 2 + 1] = l1; }
    }
  }
}
"""


def _cuda_source():
    return (_CUDA_SRC.replace("__STAGES__", str(STAGES))
            .replace("__WGMMA_SS_64__", _wgmma_ss(64)).replace("__WGMMA_SS_128__", _wgmma_ss(128))
            .replace("__WGMMA_RS_8__", _wgmma_rs(8)).replace("__WGMMA_RS_64__", _wgmma_rs(64))
            .replace("__ROWSUM__", str(CFG["ROWSUM"])).replace("__MAXTREE__", str(CFG["MAXTREE"]))
            .replace("__POLY__", str(CFG["POLY"])).replace("__EXPFN__", CFG["EXPFN"]).replace("__F16SM__", str(CFG["F16SM"])))


# ----------------------------------------------------------------------------------------------------------------
# NVRTC compile + driver-API launch (no torch extension build, no subprocess)
# ----------------------------------------------------------------------------------------------------------------
_STATE = {}


def _libcuda():
    if "cuda" not in _STATE:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuTensorMapEncodeTiled.restype = ctypes.c_int
        lib.cuLaunchKernel.restype = ctypes.c_int
        _STATE["cuda"] = lib
    return _STATE["cuda"]


def _nvrtc():
    """libnvrtc via ctypes (no extra Python deps): torch's bundled copy, then the system one."""
    if "nvrtc" in _STATE:
        return _STATE["nvrtc"]
    import glob
    import site
    cands = []
    for d in site.getsitepackages() + [os.path.dirname(os.path.dirname(torch.__file__))]:
        cands += sorted(glob.glob(os.path.join(d, "nvidia", "*", "lib", "libnvrtc.so*")), reverse=True)
        cands += sorted(glob.glob(os.path.join(d, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*")), reverse=True)
    cands += ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so", "/usr/local/cuda/lib64/libnvrtc.so"]
    for c in cands:
        if "builtins" in c or c.endswith(".alt.so") or ".alt.so" in c:
            continue
        try:
            lib = ctypes.CDLL(c)
            _STATE["nvrtc"] = lib
            return lib
        except OSError:
            pass
    raise RuntimeError("libnvrtc not found")


def _compile():
    if "func" in _STATE:
        return _STATE["func"]
    nv = _nvrtc()
    src = _cuda_source().encode()
    prog = ctypes.c_void_p()
    rc = nv.nvrtcCreateProgram(ctypes.byref(prog), ctypes.c_char_p(src), b"attn_fwd.cu", 0, None, None)
    assert rc == 0, f"nvrtcCreateProgram {rc}"
    opts = [b"--gpu-architecture=sm_90a", b"-std=c++17", b"-default-device", b"--ptxas-options=-v"]
    copts = (ctypes.c_char_p * len(opts))(*opts)
    rc = nv.nvrtcCompileProgram(prog, len(opts), copts)
    n = ctypes.c_size_t()
    nv.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    nv.nvrtcGetProgramLog(prog, log)
    _STATE["log"] = log.value.decode(errors="replace")
    if rc != 0:
        raise RuntimeError("NVRTC failed:\n" + _STATE["log"])
    nv.nvrtcGetCUBINSize(prog, ctypes.byref(n))
    cubin = ctypes.create_string_buffer(n.value)
    rc = nv.nvrtcGetCUBIN(prog, cubin)
    assert rc == 0, f"nvrtcGetCUBIN {rc}"
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    lib = _libcuda()
    torch.cuda.init()
    torch.empty(1, device="cuda")  # make sure torch's (primary) context is current
    mod = ctypes.c_void_p()
    rc = lib.cuModuleLoadData(ctypes.byref(mod), cubin)
    assert rc == 0, f"cuModuleLoadData {rc}"
    fn = ctypes.c_void_p()
    rc = lib.cuModuleGetFunction(ctypes.byref(fn), mod, b"attn_fwd")
    assert rc == 0, f"cuModuleGetFunction {rc}"
    rc = lib.cuFuncSetAttribute(fn, 8, ctypes.c_int(smem_bytes()))  # MAX_DYNAMIC_SHARED_SIZE_BYTES
    assert rc == 0, f"cuFuncSetAttribute {rc}"
    _STATE["mod"], _STATE["func"], _STATE["cubin"] = mod, fn, cubin.raw
    return fn


def smem_bytes():
    return 1024 + STAGES * (64 * BN + BN * 64) + QPB * 8 * 64 + 1024 + 8 * (2 * STAGES + 1)


class _TmaDesc(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64 * 16)]


def _tmap(ptr, dims, strides, box, swizzle):
    """uint8 tensor map. dims/box innermost first; strides (bytes) for dims[1:]."""
    lib = _libcuda()
    d = _TmaDesc()
    r = len(dims)
    gd = (ctypes.c_uint64 * r)(*dims)
    gs = (ctypes.c_uint64 * max(r - 1, 1))(*strides)
    bd = (ctypes.c_uint32 * r)(*box)
    es = (ctypes.c_uint32 * r)(*([1] * r))
    rc = lib.cuTensorMapEncodeTiled(ctypes.byref(d), 0, r, ctypes.c_void_p(ptr), gd, gs, bd, es, 0, swizzle, 2, 0)
    assert rc == 0, f"cuTensorMapEncodeTiled {rc} dims={dims} strides={strides} box={box}"
    return d


def _launch(grid, args_c):
    fn = _compile()
    lib = _libcuda()
    params = (ctypes.c_void_p * len(args_c))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args_c])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.cuLaunchKernel(fn, grid[0], grid[1], grid[2], NTHREADS, 1, 1, smem_bytes(), stream, params, None)
    assert rc == 0, f"cuLaunchKernel {rc}"


# ----------------------------------------------------------------------------------------------------------------
# Triton helpers: paged gather (K contiguous, V transposed + key-permuted) and split combine
# ----------------------------------------------------------------------------------------------------------------
@triton.jit
def _gather_kv_kernel(kc_ptr, vc_ptr, pt_ptr, seqlens_ptr, k_out, vt_out, PT_STRIDE, Spad,
                      PAGE: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    b = tl.program_id(1)
    S = tl.load(seqlens_ptr + b)
    if tile * BLOCK >= S:  # never read by the attention kernel
        return
    p = tile * BLOCK + tl.arange(0, BLOCK)
    # V^T key permutation inside each 32-key chunk: logical k = 16h + 4q + i  <-  actual 16h + 2q + (i&1) + 8(i>>1)
    k = p % 32
    h = k // 16
    r = k % 16
    act = p - k + 16 * h + 2 * (r // 4) + (r % 4) % 2 + 8 * ((r % 4) // 2)
    c = tl.arange(0, 128)
    # K (no permutation)
    ok = p < S
    pg = tl.load(pt_ptr + b * PT_STRIDE + p // PAGE, mask=ok, other=0).to(tl.int64)
    row = pg * PAGE + p % PAGE
    kv = tl.load(kc_ptr + row[:, None] * 128 + c[None, :], mask=ok[:, None], other=0)
    kv = tl.reshape(kv, (BLOCK, 2, 64))
    k0, k1 = tl.split(tl.permute(kv, (0, 2, 1)))
    d = tl.arange(0, 64)
    base0 = (b * 2 + 0).to(tl.int64) * Spad
    base1 = (b * 2 + 1).to(tl.int64) * Spad
    tl.store(k_out + (base0 + p)[:, None] * 64 + d[None, :], k0)
    tl.store(k_out + (base1 + p)[:, None] * 64 + d[None, :], k1)
    # V^T
    oka = act < S
    pga = tl.load(pt_ptr + b * PT_STRIDE + act // PAGE, mask=oka, other=0).to(tl.int64)
    rowa = pga * PAGE + act % PAGE
    vv = tl.load(vc_ptr + rowa[:, None] * 128 + c[None, :], mask=oka[:, None], other=0)
    vv = tl.reshape(vv, (BLOCK, 2, 64))
    v0, v1 = tl.split(tl.permute(vv, (0, 2, 1)))   # [BLOCK, 64] each
    vb0 = (b * 2 + 0).to(tl.int64) * 64
    vb1 = (b * 2 + 1).to(tl.int64) * 64
    tl.store(vt_out + (vb0 + d)[:, None] * Spad + p[None, :], tl.trans(v0))
    tl.store(vt_out + (vb1 + d)[:, None] * Spad + p[None, :], tl.trans(v1))


@triton.jit
def _attn_combine_kernel(opart, mlpart, vscale_ptr, out_ptr, R, NSPLIT: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = rows < R
    d = tl.arange(0, 64)
    mmax = tl.full([BLOCK_R], float("-inf"), tl.float32)
    for s in tl.static_range(NSPLIT):
        m = tl.load(mlpart + (s * R + rows) * 2, mask=rm, other=float("-inf"))
        mmax = tl.maximum(mmax, m)
    mref = tl.where(mmax == float("-inf"), 0.0, mmax)
    acc = tl.zeros([BLOCK_R, 64], tl.float32)
    lsum = tl.zeros([BLOCK_R], tl.float32)
    for s in tl.static_range(NSPLIT):
        m = tl.load(mlpart + (s * R + rows) * 2, mask=rm, other=float("-inf"))
        l = tl.load(mlpart + (s * R + rows) * 2 + 1, mask=rm, other=0.0)
        w = tl.exp2(m - mref)
        w = tl.where(m == float("-inf"), 0.0, w)
        o = tl.load(opart + (s * R + rows)[:, None].to(tl.int64) * 64 + d[None, :], mask=rm[:, None], other=0.0)
        acc += o * w[:, None]
        lsum += l * w
    vs = tl.load(vscale_ptr)
    inv = tl.where(lsum > 0, vs / lsum, 0.0)
    tl.store(out_ptr + rows[:, None].to(tl.int64) * 64 + d[None, :], (acc * inv[:, None]).to(tl.bfloat16),
             mask=rm[:, None])


# ----------------------------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------------------------
def gather_kv(k_cache, v_cache, page_table, cache_seqlens, B, Spad):
    page = k_cache.shape[1]
    Kc = torch.empty((B * 2 * Spad, 64), dtype=torch.uint8, device=k_cache.device)
    Vt = torch.empty((B * 2 * 64, Spad), dtype=torch.uint8, device=k_cache.device)
    _gather_kv_kernel[(Spad // BN, B)](k_cache.view(torch.uint8), v_cache.view(torch.uint8), page_table,
                                       cache_seqlens, Kc, Vt, page_table.stride(0), Spad,
                                       PAGE=page, BLOCK=BN, num_warps=4)
    return Kc, Vt


def pick_splits(B, L, max_keys, sms=132):
    """Split-KV only when there are too few query tiles for the SMs and the context is long: pick the split count
    minimising waves-per-split ceil(ctas * ns / sms) / ns (ties -> fewer splits), >= 32 key tiles per split."""
    ctas = triton.cdiv(L, QPB) * 2 * B
    if ctas >= 2 * sms:
        return 1
    best, best_cost = 1, float(triton.cdiv(ctas, sms))
    for ns in range(2, 9):
        if max_keys // BN // ns < 32:
            break
        cost = triton.cdiv(ctas * ns, sms) / ns
        if cost < best_cost - 1e-9:
            best, best_cost = ns, cost
    return best


def custom_attention(q8, k_cache, v_cache, page_table, cache_seqlens, L, softmax_scale, k_scale, v_scale,
                     num_splits=None):
    """Full causal attention (bottom-right aligned), uniform q_len L per request. Returns [T, 16, 64] bf16."""
    T = q8.shape[0]
    B = T // L
    page = k_cache.shape[1]
    max_keys = page_table.shape[1] * page
    Spad = triton.cdiv(max_keys, BN) * BN
    Kc, Vt = gather_kv(k_cache, v_cache, page_table, cache_seqlens, B, Spad)
    nmb = triton.cdiv(L, QPB)
    ns = pick_splits(B, L, max_keys) if num_splits is None else num_splits
    out = torch.empty((T, 16, 64), dtype=torch.bfloat16, device=q8.device)
    if ns > 1:
        opart = torch.empty((ns, T, 16, 64), dtype=torch.float32, device=q8.device)
        mlpart = torch.empty((ns, T, 16, 2), dtype=torch.float32, device=q8.device)
    else:
        opart = mlpart = out  # unused
    tmK = _tmap(Kc.data_ptr(), [64, B * 2 * Spad], [64], [64, BN], 2)
    tmV = _tmap(Vt.data_ptr(), [Spad, B * 2 * 64], [Spad], [BN, 64], 3)
    qv = q8.view(torch.uint8)
    tmQ = _tmap(qv.data_ptr(), [64, 8, 2, T], [64, 512, 1024], [64, 8, 1, QPB], 2)
    args = [tmK, tmV, tmQ,
            ctypes.c_void_p(cache_seqlens.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(opart.data_ptr()), ctypes.c_void_p(mlpart.data_ptr()),
            ctypes.c_void_p(k_scale.data_ptr()), ctypes.c_void_p(v_scale.data_ptr()),
            ctypes.c_int(L), ctypes.c_int(T), ctypes.c_int(Spad), ctypes.c_int(ns),
            ctypes.c_float(softmax_scale * LOG2E)]
    _launch((nmb, 2 * ns, B), args)
    if ns > 1:
        R = T * 16
        _attn_combine_kernel[(triton.cdiv(R, 64),)](opart, mlpart, v_scale, out, R, NSPLIT=ns, BLOCK_R=64, num_warps=4)
    # keep the scratch alive until the kernels are enqueued (caching allocator is stream-ordered)
    return out


def use_custom(window, L, B, max_keys):
    """Custom kernel for full-attention layers with enough work (all five scored full shapes: >= 6e7 q*k pairs per
    head); FA3 for sliding-window layers and small blocks (there FA3 is equal or faster)."""
    return window == 0 and B * L * max_keys >= 3e7


def attention(q8, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, L, softmax_scale, window,
              k_scale, v_scale, kv_scale_is_one=True, scheduler_metadata=None):
    B = q8.shape[0] // L
    if use_custom(window, L, B, page_table.shape[1] * k_cache.shape[1]):
        return custom_attention(q8, k_cache, v_cache, page_table, cache_seqlens, L, softmax_scale,
                                k_scale.reshape(1).float(), v_scale.reshape(1).float())
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    descale = {}
    if not kv_scale_is_one:
        descale = dict(k_descale=k_scale.reshape(1, 1).expand(B, 2), v_descale=v_scale.reshape(1, 1).expand(B, 2))
    out = flash_attn_with_kvcache(q=q8, k_cache=k_cache, v_cache=v_cache, page_table=page_table,
                                  cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=L,
                                  softmax_scale=softmax_scale, causal=True,
                                  window_size=(window - 1, 0) if window else (-1, -1),
                                  scheduler_metadata=scheduler_metadata, **descale)
    return out[0] if isinstance(out, tuple) else out


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
def _attn_boundary_kernel(a_ptr, s_ptr, alpha_ptr, beta_ptr, w_ptr, s1_ptr, h_ptr, N: tl.constexpr, CH: tl.constexpr):
    """stream1 = alpha*s + beta*rms(a); h = rms(stream1)*(1+w). Row-chunked: 3 passes over the row (the re-reads
    hit L2); measured 3.9 TB/s on H200 vs 2.9 for the whole-row-in-registers version."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, CH)
    acc = tl.zeros([CH], tl.float32)
    for c in tl.static_range(N // CH):
        a = tl.load(a_ptr + row * N + c * CH + offs).to(tl.float32)
        acc += a * a
    ra = tl.rsqrt(tl.sum(acc, 0) / N + 1e-8)
    acc2 = tl.zeros([CH], tl.float32)
    for c in tl.static_range(N // CH):
        o = c * CH + offs
        a = tl.load(a_ptr + row * N + o).to(tl.float32)
        s = tl.load(s_ptr + row * N + o)
        s1 = tl.load(alpha_ptr + o) * s + tl.load(beta_ptr + o) * (a * ra)
        tl.store(s1_ptr + row * N + o, s1)
        acc2 += s1 * s1
    rs = tl.rsqrt(tl.sum(acc2, 0) / N + 1e-5)
    for c in tl.static_range(N // CH):
        o = c * CH + offs
        s1 = tl.load(s1_ptr + row * N + o)
        tl.store(h_ptr + row * N + o, (s1 * rs * tl.load(w_ptr + o)).to(tl.bfloat16))


@triton.jit
def _ffn_boundary_kernel(f_ptr, s1_ptr, wf_ptr, alpha_ptr, beta_ptr, wn_ptr, s2_ptr, hid_ptr, N: tl.constexpr,
                         CH: tl.constexpr):
    """stream2 = alpha*stream1 + beta*rms(f*(1+wf)); hidden = rms(stream2)*(1+wn). Row-chunked like above."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, CH)
    acc = tl.zeros([CH], tl.float32)
    for c in tl.static_range(N // CH):
        o = c * CH + offs
        f = tl.load(f_ptr + row * N + o).to(tl.float32) * tl.load(wf_ptr + o)
        acc += f * f
    rf = tl.rsqrt(tl.sum(acc, 0) / N + 1e-8)
    acc2 = tl.zeros([CH], tl.float32)
    for c in tl.static_range(N // CH):
        o = c * CH + offs
        f = tl.load(f_ptr + row * N + o).to(tl.float32) * tl.load(wf_ptr + o)
        s1 = tl.load(s1_ptr + row * N + o)
        s2 = tl.load(alpha_ptr + o) * s1 + tl.load(beta_ptr + o) * (f * rf)
        tl.store(s2_ptr + row * N + o, s2)
        acc2 += s2 * s2
    r2 = tl.rsqrt(tl.sum(acc2, 0) / N + 1e-5)
    for c in tl.static_range(N // CH):
        o = c * CH + offs
        s2 = tl.load(s2_ptr + row * N + o)
        tl.store(hid_ptr + row * N + o, (s2 * r2 * tl.load(wn_ptr + o)).to(tl.bfloat16))


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


SPLIT_ROWS = 32  # experts with more routes than this run in the large-tile launch (H200 sweep on real routing)


def _tuning_big(tokens):
    return _gemm_cfg(160 if tokens <= 1024 else 176, 256, 64)


def _tuning_small(tokens):
    return _gemm_cfg(16, 256, 128)


@triton.jit
def _split_counts_kernel(cnt_ptr, big_ptr, small_ptr, THR: tl.constexpr):
    e = tl.arange(0, 32)
    c = tl.load(cnt_ptr + e)
    tl.store(big_ptr + e, tl.where(c > THR, c, 0))
    tl.store(small_ptr + e, tl.where(c > THR, 0, c))


def _humming_run(bank, x, out, counts, tuning, valid_m, locks="locks"):
    from humming import ops
    t = bank["tensors"]
    return ops.humming_gemm(
        layer_config=bank["config"], compute_config=_CC, tuning_config=json.dumps(tuning),
        inputs=x, weight=t["weight"], weight_scale=t["weight_scale"], weight_scale_2=t.get("weight_scale_2"),
        outputs=out, expert_layout=counts, locks=bank[locks], top_k=1, valid_shape_m=valid_m)


def _experts_gemm(st, name, x, out, cb, cs, tb, ts, valid_m):
    """Large-expert launch on the current stream and small-expert launch on the side stream, concurrently (the
    small launch is memory-bound and fills the large one's tail: -2..5%). Separate stream-K lock buffers."""
    main, side = torch.cuda.current_stream(), st["side"]
    side.wait_stream(main)
    _humming_run(st[name], x, out, cb, tb, valid_m)
    with torch.cuda.stream(side):
        _humming_run(st[name], x, out, cs, ts, valid_m, locks="locks2")
    main.wait_stream(side)


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
    for name in ("w13", "w2"):
        st[name]["locks"] = torch.zeros(1024, dtype=torch.int32, device=dev)
        st[name]["locks2"] = torch.zeros(1024, dtype=torch.int32, device=dev)
    st["cnt"] = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    st["cnt_big"] = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    st["cnt_small"] = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    st["side"] = torch.cuda.Stream(dev)
    _buffers(st, 16384, dev)
    _compile()  # NVRTC-compile the attention kernel (~1 s)
    # Pre-build (JIT) the expert GEMM kernels for the token counts we expect.
    from humming.kernel.humming import HummingKernel
    for name in ("w13", "w2"):
        for tuning in (_tuning_big(0), _tuning_big(1 << 20), _tuning_small(0)):
            HummingKernel.prepare_kernels(st[name]["config"], _CC, json.dumps(tuning))
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
    out = attention(q8, kc, vc, data["page_table"], data["cache_seqlens"], data["cu_seqlens_q"], length,
                    cfg["sm_scale"] * cfg["qk_scale_factor"], window, data["k_scale"], data["v_scale"],
                    kv_scale_is_one=cfg["kv_scale"] == 1.0, scheduler_metadata=data["scheduler_metadata"])
    o_in = torch.empty((T, 1024), dtype=torch.bfloat16, device=dev)
    _onorm_gate_kernel[(triton.cdiv(T, BT),)](out, qkv, o_in, T, QKV_STRIDE=2304, BLOCK_T=BT, num_warps=4)
    a = torch.mm(o_in, st["o"].t())  # [T, 8192] bf16

    # ---- post-attention boundary ----
    s1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    h = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _attn_boundary_kernel[(T,)](a, data["stream"], st["a1"], st["b1"], st["w_pre_ffn"], s1, h, N=HIDDEN,
                                CH=4096, num_warps=8)

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
    cb, cs = st["cnt_big"], st["cnt_small"]
    _split_counts_kernel[(1,)](cnt, cb, cs, THR=SPLIT_ROWS, num_warps=1)
    tb, ts = _tuning_big(T), _tuning_small(T)
    _experts_gemm(st, "w13", xg, h13, cb, cs, tb, ts, valid_m)
    _silu_mul_kernel[(T,)](h13, rowmap, inter, N=D_EXPERT, num_warps=8)
    _experts_gemm(st, "w2", inter, y, cb, cs, tb, ts, valid_m)
    moe = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    _combine_kernel[(T,)](y, rowmap, tw, st["w_post_exp"], moe, N=D_EXPERT, num_warps=8)

    # ---- unpooler + FFN boundary ----
    f = torch.mm(moe, st["unpooler"].t())  # [T, 8192]
    s2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hid = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    _ffn_boundary_kernel[(T,)](f, s1, st["w_post_ffn"], st["a2"], st["b2"], st["w_next"], s2, hid, N=HIDDEN,
                               CH=4096, num_warps=8)
    return hid, s2, ids
