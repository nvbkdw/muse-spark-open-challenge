# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open: skinny GEMMs on losslessly compressed BF16 weights.

Small-m path (m <= 32): one CUDA kernel per call.
  * Weights are repacked once in prepare() into mma.sync m16n8k16 A-fragment order and
    losslessly compressed to 12 bits/weight: the sign+mantissa byte is kept verbatim and the
    8-bit exponent becomes a 4-bit code n = E - base for the 16-exponent window that holds
    almost every weight.  Weights outside the window (about 1e-4 of them, tiny or zero) are
    stored as code 0 and fixed up exactly from a sparse (row, k, delta) list.  Decoding is
    exact: a bf16 with exponent 112+n is the original weight times 2^(112-base); the fp32
    result is multiplied by 2^(base-112), a power of two, at the end.
  * The router keeps its FP32 weights as an interleaved BF16 hi/lo pair (k-major [hi_k, lo_k])
    with FP32 accumulation, as SGLang does above 8 tokens; x is duplicated in registers.
  * Swap-AB: W tile is the MMA A operand (16 rows), x^T the B operand (8 tokens).
  * A producer warp streams contiguous weight stages and x slices into shared memory with
    1-D TMA bulk copies (mbarrier pipeline); 8 consumer warps decode + MMA.
  * Work is split stream-K over (row group, k block) superunits; partial row-group sums are
    combined with fp32 reductions in a zero-initialised workspace and the last arriving CTA
    writes the output (and re-zeroes the workspace and counter for the next replay).
Large m: cuBLAS on the original weights.
"""

import os
from typing import TypedDict

import torch
import torch.nn.functional as F


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef PFD
#define PFD 4
#endif
#ifndef DINIT
#define DINIT 2
#endif

struct Params {
  const uint8_t* w;         // packed stages: [S][STAGE_W + EXB]
  const __nv_bfloat16* x;   // [m, K]
  void* out;                // [m, N]
  float* ws;                // [N, 32] fp32, zero between calls
  int* cnt;                 // [row groups], zero between calls
  int m, N, K;              // K: actual x row length
  int SPR, S;               // superunits per row group, total superunits
  float scale;
  unsigned long long* tdbg;
};

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" :: "r"(smem_u32(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_1d_hint(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
      :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 12-bit decode of two weights: byte pair -> [b1 b1 b0 b0], keep sign+mantissa, insert 4-bit exponent code + 112
__device__ __forceinline__ uint32_t dec(uint32_t bw, uint32_t sel, uint32_t nsh) {
  uint32_t r = __byte_perm(bw, 0, sel);
  uint32_t t, o;
  // t = (r & 0x807F807F) | (nsh & ~0x807F807F)   (bitwise select)
  asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(t) : "r"(r), "r"(nsh), "r"(0x807F807Fu));
  // o = (t & 0x87FF87FF) | 0x38003800
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(o) : "r"(t), "r"(0x87FF87FFu), "r"(0x38003800u));
  return o;
}
__device__ __forceinline__ void bar_consumers(int nthreads) { asm volatile("bar.sync 1, %0;" :: "r"(nthreads) : "memory"); }
__device__ __forceinline__ void red_add_v2(float* p, float a, float b) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }

template <bool OUTF32>
__device__ __forceinline__ void store_out(void* out, size_t idx, float v) {
  if (OUTF32) reinterpret_cast<float*>(out)[idx] = v;
  else reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(v);
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ int2 ldg_i2(const void* p) {
  int2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

// W: 16-row tiles (= warps) per row group; KB: 64-wide (virtual) k chunks per superunit;
// MB: 8-token blocks; D: register prefetch depth (units); RAW: plain bf16 fragments (router hi/lo).
template <int W, int KB, int MB, int D, bool RAW, bool DUPX, bool OUTF32>
__global__ void __launch_bounds__(W * 32, 1) skinny_kernel(const Params p) {
  constexpr int NP = RAW ? 4 : 3;               // 16-byte parts per lane per unit
  constexpr int UB = NP * 512;
  constexpr int EXB = RAW ? 0 : 256;
  constexpr int SU_BYTES = W * KB * UB + EXB;
  constexpr int XCH = DUPX ? 64 : 128;          // x bytes per token per chunk
  constexpr int MT = MB * 8;
  constexpr int ROWS = W * 16;
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* xbar = reinterpret_cast<uint64_t*>(smem);  // [2]
  uint8_t* xs = smem + 128;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int G = gridDim.x, bid = blockIdx.x;
  const int S = p.S, SPR = p.SPR;
  const int s0 = (int)(((long long)bid * S) / G);
  const int s1 = (int)(((long long)(bid + 1) * S) / G);
  const int n = s1 - s0;
  const int m = p.m;
  const int nmax = (S + G - 1) / G;
  const int XROW = nmax * KB * XCH + 16;        // bytes per token row (16 mod 128 -> conflict-free)
  const int nq = n * KB;
#ifdef TIMING
  if (threadIdx.x == 0) { p.tdbg[bid * 8 + 0] = gtime(); unsigned sm; asm volatile("mov.u32 %0, %smid;" : "=r"(sm)); p.tdbg[bid * 8 + 6] = sm; }
#endif
  // ---- 1) x for this CTA's k-range -> shared memory (TMA), issued first; barrier 0 = first superunit ----
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(xbar)));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(xbar + 1)));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0) {
    __syncwarp();
    const int rgA = s0 / SPR, rgB = (s1 - 1) / SPR;
    if (lane == 0) {
      mbar_expect_tx(xbar, (uint32_t)(m * KB * XCH));
      mbar_expect_tx(xbar + 1, (uint32_t)(m * (n - 1) * KB * XCH));
    }
    __syncwarp();
    for (int rg = rgA; rg <= rgB; ++rg) {
      int a = rg == rgA ? s0 : rg * SPR;
      const int b = rg == rgB ? s1 : (rg + 1) * SPR;
      if (rg == rgA) {
        for (int tok = lane; tok < m; tok += 32)
          tma_load_1d(xs + tok * XROW, reinterpret_cast<const uint8_t*>(p.x) + (size_t)tok * p.K * 2 + (size_t)(a - rg * SPR) * KB * XCH,
                      KB * XCH, xbar);
        ++a;
      }
      if (b > a) {
        const uint32_t bytes = (uint32_t)((b - a) * KB * XCH);
        const size_t goff = (size_t)(a - rg * SPR) * KB * XCH;
        for (int tok = lane; tok < m; tok += 32)
          tma_load_1d(xs + tok * XROW + (a - s0) * KB * XCH,
                      reinterpret_cast<const uint8_t*>(p.x) + (size_t)tok * p.K * 2 + goff, bytes, xbar + 1);
      }
    }
  }
  // ---- 2) prefetch the first D weight units into registers ----
  const uint8_t* wbase = p.w + (size_t)s0 * SU_BYTES + warp * UB + lane * 16;
  auto uaddr = [&](int q) -> const uint8_t* {
    const int it = q / KB, j = q - it * KB;
    return wbase + (size_t)it * SU_BYTES + j * W * UB;
  };
  uint4 buf[D][NP];
  int2 ebuf[D];
#pragma unroll
  for (int d = 0; d < (DINIT < D ? DINIT : D); ++d) {
    if (d < nq) {
      const uint8_t* u = uaddr(d);
#pragma unroll
      for (int pp = 0; pp < NP; ++pp) buf[d][pp] = ldg_stream(u + pp * 512);
      if (!RAW && (d % KB) == 0) ebuf[d] = ldg_i2(p.w + (size_t)(s0 + d / KB) * SU_BYTES + W * KB * UB + lane * 8);
    }
  }
  __syncthreads();

  const int g = lane >> 2, t = lane & 3;
  float acc[MB][4];
#pragma unroll
  for (int i = 0; i < MB; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  int cur_rg = s0 / SPR;

  auto flush = [&](int rg) {
    const float sc = p.scale;
    const int row0 = rg * ROWS + warp * 16 + g;
    const int first = rg * SPR, last = rg * SPR + SPR - 1;
    const bool whole = (first >= s0) && (last < s1);
    if (whole) {
#pragma unroll
      for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int tok = mb * 8 + 2 * t + (e & 1);
          if (tok < m) store_out<OUTF32>(p.out, (size_t)tok * p.N + row0 + (e >> 1) * 8, acc[mb][e] * sc);
        }
    } else {
      const int ca = (int)(((long long)(first + 1) * G - 1) / S);
      const int cl = (int)(((long long)(last + 1) * G - 1) / S);
      if (bid != ca) {
#pragma unroll
        for (int mb = 0; mb < MB; ++mb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int tok = mb * 8 + 2 * t;
            if (tok < m) red_add_v2(p.ws + (size_t)(row0 + h * 8) * 32 + tok, acc[mb][h * 2] * sc, acc[mb][h * 2 + 1] * sc);
          }
        bar_consumers(W * 32);
        if (threadIdx.x == 0) {
          asm volatile("fence.acq_rel.gpu;" ::: "memory");
          asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;" :: "l"(p.cnt + rg) : "memory");
        }
      } else {
        if (threadIdx.x == 0) {
          const int need = cl - ca;
          int v;
          do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p.cnt + rg) : "memory"); } while (v < need);
        }
        bar_consumers(W * 32);
#pragma unroll
        for (int mb = 0; mb < MB; ++mb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int tok = mb * 8 + 2 * t;
            if (tok < m) {
              const int row = row0 + h * 8;
              float* wp = p.ws + (size_t)row * 32 + tok;
              const float2 o = __ldcg(reinterpret_cast<const float2*>(wp));
              *reinterpret_cast<float2*>(wp) = make_float2(0.f, 0.f);
              store_out<OUTF32>(p.out, (size_t)tok * p.N + row, o.x + acc[mb][h * 2] * sc);
              if (tok + 1 < m) store_out<OUTF32>(p.out, (size_t)(tok + 1) * p.N + row, o.y + acc[mb][h * 2 + 1] * sc);
            }
          }
        if (threadIdx.x == 0) p.cnt[rg] = 0;
      }
    }
#pragma unroll
    for (int i = 0; i < MB; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  };

  mbar_wait(xbar, 0);
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 2] = gtime();
#endif
#pragma unroll
  for (int d = (DINIT < D ? DINIT : D); d < D; ++d) {
    if (d < nq) {
      const uint8_t* u = uaddr(d);
#pragma unroll
      for (int pp = 0; pp < NP; ++pp) buf[d][pp] = ldg_stream(u + pp * 512);
      if (!RAW && (d % KB) == 0) ebuf[d] = ldg_i2(p.w + (size_t)(s0 + d / KB) * SU_BYTES + W * KB * UB + lane * 8);
    }
  }

  for (int q0 = 0; q0 < nq; q0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int q = q0 + d;
      if (q < nq) {
        const int it = q / KB, j = q - it * KB;
        if (q == KB) mbar_wait(xbar + 1, 0);
        if (j == 0) {
          const int rg = (s0 + it) / SPR;
          if (rg != cur_rg) { flush(cur_rg); cur_rg = rg; }
        }
        uint32_t a[4][4];
        if (RAW) {
#pragma unroll
          for (int s = 0; s < 4; ++s) { a[s][0] = buf[d][s].x; a[s][1] = buf[d][s].y; a[s][2] = buf[d][s].z; a[s][3] = buf[d][s].w; }
        } else {
          const uint4 p0 = buf[d][0], p1 = buf[d][1], p2 = buf[d][2];
          const uint32_t bw[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
          const uint32_t nw[4] = {p2.x, p2.y, p2.z, p2.w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            a[s][0] = dec(bw[2 * s], 0x1100, nw[s] << 7);
            a[s][1] = dec(bw[2 * s], 0x3322, nw[s] << 3);
            a[s][2] = dec(bw[2 * s + 1], 0x1100, nw[s] >> 1);
            a[s][3] = dec(bw[2 * s + 1], 0x3322, nw[s] >> 5);
          }
        }
        int2 ex = make_int2(0, 0);
        if (!RAW && j == 0) ex = ebuf[d];
        // refill this slot with unit q + D
        if (q + D < nq) {
          const uint8_t* u = uaddr(q + D);
#pragma unroll
          for (int pp = 0; pp < NP; ++pp) buf[d][pp] = ldg_stream(u + pp * 512);
          if (!RAW && ((q + D) % KB) == 0) ebuf[d] = ldg_i2(p.w + (size_t)(s0 + (q + D) / KB) * SU_BYTES + W * KB * UB + lane * 8);
        }
        const uint8_t* xq = xs + q * XCH;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          const uint8_t* xr = xq + (mb * 8 + g) * XROW;
          uint32_t b[8];
          if (DUPX) {
            const uint4 v = lds128(xr + t * 16);
            const uint32_t xw[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int s = 0; s < 4; ++s) { b[2 * s] = __byte_perm(xw[s], 0, 0x1010); b[2 * s + 1] = __byte_perm(xw[s], 0, 0x3232); }
          } else {
            const uint4 v0 = lds128(xr + t * 32), v1 = lds128(xr + t * 32 + 16);
            b[0] = v0.x; b[1] = v0.y; b[2] = v0.z; b[3] = v0.w; b[4] = v1.x; b[5] = v1.y; b[6] = v1.z; b[7] = v1.w;
          }
#pragma unroll
          for (int s = 0; s < 4; ++s) mma16816(acc[mb], a[s][0], a[s][1], a[s][2], a[s][3], b[2 * s], b[2 * s + 1]);
        }
        if (!RAW && j == 0) {
          // exact fix-up of this superunit's out-of-window weights: lane 0 holds the count, lane e entry e
          const int ne = __shfl_sync(0xffffffffu, ex.x, 0);
          for (int e = 1; e <= ne; ++e) {
            const int meta = __shfl_sync(0xffffffffu, ex.x, e);
            const float dlt = __int_as_float(__shfl_sync(0xffffffffu, ex.y, e));
            const int row = meta >> 16;
            if ((row >> 4) == warp && (row & 7) == g) {
              const int kl = meta & 0xFFFF;
              const bool hi = (row >> 3) & 1;
              const uint8_t* xe = xs + it * KB * XCH + kl * 2;
#pragma unroll
              for (int mb = 0; mb < MB; ++mb) {
                const int tok = mb * 8 + 2 * t;
                const float x0 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(xe + tok * XROW));
                const float x1 = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(xe + (tok + 1) * XROW));
                if (hi) { acc[mb][2] += x0 * dlt; acc[mb][3] += x1 * dlt; }
                else { acc[mb][0] += x0 * dlt; acc[mb][1] += x1 * dlt; }
              }
            }
          }
        }
      }
    }
  }
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 4] = gtime();
#endif
  if (n > 0) flush(cur_rg);
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 5] = gtime();
#endif
}

template <int W, int KB, int MB, int D, bool RAW, bool DUPX, bool OUTF32>
static void launch_t(const Params& p, int grid) {
  constexpr int XCH = DUPX ? 64 : 128;
  const int nmax = (p.S + grid - 1) / grid;
  const size_t smem = 128 + (size_t)MB * 8 * (nmax * KB * XCH + 16);
  auto kern = skinny_kernel<W, KB, MB, D, RAW, DUPX, OUTF32>;
  static int cur = 0;
  if ((int)smem > cur) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cur = (int)smem;
  }
  kern<<<grid, W * 32, smem, at::cuda::getCurrentCUDAStream()>>>(p);
}

template <int W, int KB, int D, bool RAW, bool DUPX, bool OUTF32>
static void launch_mb(const Params& p, int grid, int mb) {
  switch (mb) {
    case 1: launch_t<W, KB, 1, D, RAW, DUPX, OUTF32>(p, grid); break;
    case 2: launch_t<W, KB, 2, D, RAW, DUPX, OUTF32>(p, grid); break;
    case 3: launch_t<W, KB, 3, D, RAW, DUPX, OUTF32>(p, grid); break;
    default: launch_t<W, KB, 4, D, RAW, DUPX, OUTF32>(p, grid); break;
  }
}

void skinny(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
            int64_t N, int64_t SPR, int64_t grid, double scale, int64_t mode, torch::Tensor tdbg) {
  Params p;
  p.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.out = out.data_ptr();
  p.ws = ws.data_ptr<float>();
  p.cnt = cnt.data_ptr<int>();
  p.m = (int)x.size(0);
  p.K = (int)x.size(1);
  p.N = (int)N;
  p.SPR = (int)SPR;
  const int rows_per_rg = (mode == 2) ? 256 : 128;
  p.S = (int)((N / rows_per_rg) * SPR);
  p.scale = (float)scale;
  p.tdbg = reinterpret_cast<unsigned long long*>(tdbg.data_ptr());
  const int mb = (p.m + 7) / 8;
  int g = (int)grid;
  if (g > p.S) g = p.S;
  if (mode == 0) launch_mb<8, 2, PFD, false, false, false>(p, g, mb);
  else if (mode == 2) launch_mb<16, 1, 4, false, false, false>(p, g, mb);
  else launch_mb<8, 2, 4, true, true, true>(p, g, mb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("skinny", &skinny); }
"""

_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        defs = os.environ.get("SKINNY_DEFS", "").split()
        _ext = load_inline(
            name="muse_proj_skinny_v2" + "".join("_" + "".join(c for c in d if c.isalnum()) for d in defs),
            cpp_sources="",
            cuda_sources=CUDA_SRC,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"] + defs,
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _ext


W_TILES = 8      # 16-row tiles per row group (= consumer warps)
KB = 2           # 64-wide k chunks per pipeline stage
_cfg_env = os.environ.get("SKINNY_CFG", "")
# (layer, variant) -> (tiles per row group, chunks per stage, kernel mode)
LAYER_CFG = {}
if _cfg_env == "16":
    LAYER_CFG = {(l, v): (16, 1, 2) for l in ("qkv_gate", "o", "pooler", "unpooler") for v in ("full", "slice")}


def _frag_order(w, n_rows, kv, wt=8):
    """[N, Kv] -> [RG, CH, W, 4(s), 8(g), 4(t), 2(qhi), 2(h), 2(qlo)] element order."""
    RG, CH = n_rows // (16 * wt), kv // 64
    v = w.reshape(RG, wt, 2, 8, CH, 4, 4, 2, 2)  # rg, tile, h, g, ch, t, s, qhi, qlo
    return v.permute(0, 4, 1, 6, 3, 5, 7, 2, 8)        # rg, ch, tile, s, g, t, qhi, h, qlo


def _pack_raw(w):
    n_rows, kv = w.shape
    v = _frag_order(w, n_rows, kv).contiguous()
    return v.view(torch.uint8).reshape(-1)


EXB_INTS = 64     # 256-byte exception block per superunit: [count, pad, (meta, delta) x 31]
EXB_MAX = 31


def _pack_compressed(w, W_TILES=8, KB=2):
    """Lossless 12-bit packing into [superunit][weights 24 KiB | exception block 256 B].

    Returns (packed uint8, scale) or None when the weights do not fit the format."""
    n_rows, K = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    E = (bits >> 7) & 0xFF
    hist = torch.bincount(E.reshape(-1), minlength=256)
    win = torch.stack([hist[b:b + 16].sum() for b in range(0, 241)])
    base = int(win.argmax())
    inwin = (E >= base) & (E <= base + 15)
    sm = ((bits >> 8) & 0x80) | (bits & 0x7F)
    sm = torch.where(inwin, sm, torch.zeros_like(sm))
    nib = torch.where(inwin, E - base, torch.zeros_like(E))
    RG, CH = n_rows // (16 * W_TILES), K // 64
    SPR = CH // KB
    S = RG * SPR
    smo = _frag_order(sm, n_rows, K, W_TILES)  # rg, ch, tile, s, g, t, qhi, h, qlo
    smo = smo.reshape(RG, CH, W_TILES, 2, 2, 8, 4, 2, 2, 2).permute(0, 1, 2, 3, 5, 6, 4, 7, 8, 9)
    smb = smo.to(torch.uint8).reshape(RG, CH, W_TILES, 2, 512)
    nb = _frag_order(nib, n_rows, K, W_TILES)
    nb = nb.permute(0, 1, 2, 4, 5, 3, 8, 6, 7)  # rg, ch, tile, g, t, s, qlo, qhi, h
    nb = nb.reshape(RG, CH, W_TILES, 32, 4, 8).to(torch.int64)
    shifts = torch.arange(8, device=w.device, dtype=torch.int64) * 4
    words = (nb << shifts).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    nbytes = words.contiguous().view(torch.uint8).reshape(RG, CH, W_TILES, 1, 512)
    wpart = torch.cat([smb, nbytes], dim=3).reshape(S, -1)  # [S, KB*W*1536]
    # exceptions -> per-superunit blocks
    rows, ks = (~inwin).nonzero(as_tuple=True)
    blocks = torch.zeros((S, EXB_INTS), dtype=torch.int32, device=w.device)
    if rows.numel():
        su = (rows // (16 * W_TILES)) * SPR + ks // (64 * KB)
        order = torch.argsort(su, stable=True)
        rows, ks, su = rows[order], ks[order], su[order]
        counts = torch.bincount(su, minlength=S)
        if int(counts.max()) > EXB_MAX:
            return None
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(su.numel(), device=w.device) - start[su]
        wt = w[rows, ks].double()
        delta = ((wt - 2.0 ** (base - 127)) * 2.0 ** (112 - base)).float()
        meta = ((rows % (16 * W_TILES)) << 16) | (ks % (64 * KB))
        blocks[:, 0] = counts.to(torch.int32)
        blocks[su, 2 + 2 * rank] = meta.to(torch.int32)
        blocks[su, 3 + 2 * rank] = delta.view(torch.int32)
    packed = torch.cat([wpart, blocks.view(torch.uint8)], dim=1).reshape(-1).contiguous()
    return packed, 2.0 ** (base - 112)


def prepare(weights: Weights):
    dev = weights["qkv_gate"].device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    ext = _load_ext()
    st = {"raw": {k: v.contiguous() for k, v in weights.items()}, "nsm": nsm}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["router_hi_t"] = hi.t().contiguous()
    st["router_lo_t"] = lo.t().contiguous()
    st["router_hilo_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 512]
    packs = {}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        w = weights[name].contiguous()
        variants = {"full": w}
        if name == "unpooler":
            variants["slice"] = w[:1024].contiguous()
        for vname, wv in variants.items():
            wt, kb, mode = LAYER_CFG.get((name, vname), (8, 2, 0))
            res = _pack_compressed(wv, wt, kb)
            if res is None:
                packs[(name, vname)] = ("raw", None, 1.0, wv.shape[0], 0, 0)
            else:
                pk, sc = res
                packs[(name, vname)] = ("cmp", pk, sc, wv.shape[0], wv.shape[1] // 64 // kb, mode)
    # router: virtual K' = 2K, interleaved [hi_k, lo_k]
    rv = torch.stack([hi, lo], dim=2).reshape(256, -1).contiguous()
    packs[("router", "full")] = ("rtr", _pack_raw(rv), 1.0, 256, rv.shape[1] // 64 // KB, 1)
    st["packs"] = packs
    st["ws"] = torch.zeros(8192 * 32, dtype=torch.float32, device=dev)
    st["cnt"] = torch.zeros(1024, dtype=torch.int32, device=dev)
    st["ext"] = ext
    st["tdbg"] = torch.zeros(132 * 8 * 4, dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    return st


def kernel_fn(data: Inputs, st) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    if m <= 32:
        if layer == "unpooler" and n == 1024:
            key = (layer, "slice")
        else:
            key = (layer, "full")
        kind, pk, sc, N, SPR, mode = st["packs"][key]
        if kind == "rtr":
            out = torch.empty((m, N), dtype=torch.float32, device=x.device)
            st["ext"].skinny(pk, x, out, st["ws"], st["cnt"], N, SPR, st["nsm"], 1.0, 1, st["tdbg"])
            return out
        if kind == "cmp":
            out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
            st["ext"].skinny(pk, x, out, st["ws"], st["cnt"], N, SPR, st["nsm"], sc, mode, st["tdbg"])
            return out
    raw = st["raw"]
    if layer != "router":
        return F.linear(x, raw[layer][:n])
    if m <= 8:
        return F.linear(x.float(), raw["router"])
    acc = torch.mm(x, st["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, st["router_lo_t"], out_dtype=torch.float32)
    return acc
