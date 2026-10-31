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

struct Params {
  const uint8_t* w;         // packed fragments [rg][chunk][tile][unit]
  const __nv_bfloat16* x;   // [m, K]
  void* out;                // [m, N]
  const int* exc_off;       // [grid + 1] exception CSR per CTA
  const int4* exc;          // {row in row group, k (actual), delta bits, 0}
  int m, N, K;              // K: actual x row length
  int CPR;                  // 64-wide (virtual) chunks per row
  float scale;
  unsigned long long* tdbg;
};

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt)); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory"); }
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
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 12-bit decode of two weights: byte pair -> [b1 b1 b0 b0], keep sign+mantissa, insert 4-bit exponent code + 112
__device__ __forceinline__ uint32_t dec(uint32_t bw, uint32_t sel, uint32_t nsh) {
  uint32_t r = __byte_perm(bw, 0, sel);
  uint32_t t, o;
  asm("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(t) : "r"(r), "r"(nsh), "r"(0x807F807Fu));          // select
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(o) : "r"(t), "r"(0x87FF87FFu), "r"(0x38003800u));  // (t & m) | c
  return o;
}
__device__ __forceinline__ void bar_sync_id(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
template <bool OUTF32>
__device__ __forceinline__ void store_out(void* out, size_t idx, float v) {
  if (OUTF32) reinterpret_cast<float*>(out)[idx] = v;
  else reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16_rn(v);
}

// One cluster of CL CTAs per row group of WT 16-row tiles; the CTAs split K (DSMEM push reduction).
// In a CTA, WT*WK compute warps: warp (wt, wk) owns tile wt and every WK-th 64-wide chunk.
// Warp NW streams x through an NSLOT-deep TMA ring (XK chunks per slot); warp NW+1 gathers the exact
// corrections for out-of-window weights.  MB: 8-token blocks, D: per-warp register prefetch depth.
template <int WT, int WK, int MB, int D, int CL, int XK, int NSLOT, bool RAW>
__global__ void __launch_bounds__((WT * WK + 2) * 32, 1) skinny_kernel(const Params p) {
  constexpr bool DUPX = RAW, OUTF32 = RAW;
  constexpr int NW = WT * WK;
  constexpr int NP = RAW ? 4 : 3;               // 16-byte parts per lane per unit
  constexpr int UB = NP * 512;
  constexpr int XCH = DUPX ? 64 : 128;          // x bytes per token per chunk
  constexpr int MT = MB * 8;
  constexpr int ROWS = WT * 16;
  constexpr int RO = (ROWS + CL - 1) / CL;
  constexpr int MAXE = 128;
  constexpr int XFIRST = 1;                     // x slots issued before the weight prefetch starts
  static_assert(XK % WK == 0, "XK % WK");
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + NSLOT;
  uint64_t* excbar = empty + NSLOT;
  float* exc_acc = reinterpret_cast<float*>(smem + 256);           // [ROWS][MT] exact corrections
  float* red = exc_acc + ROWS * MT;                                // [WK][ROWS][MT]
  float* pbuf = red + (WK > 1 ? WK : 0) * ROWS * MT;               // [CL][RO][MT] (CL > 1)
  uint8_t* xs = reinterpret_cast<uint8_t*>(pbuf + (CL > 1 ? CL * RO * MT : 0));
  xs = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(xs) + 127) & ~uintptr_t(127));

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int bid = blockIdx.x;
  const int rg = bid / CL, rank = bid % CL;
  const int CPR = p.CPR;
  const int c0 = (rank * CPR) / CL, c1 = ((rank + 1) * CPR) / CL;
  const int nch = c1 - c0;                      // chunks for this CTA (multiple of XK)
  const int nslots = nch / XK;
  const int m = p.m;
  const int XROW = XK * XCH + 16;               // smem bytes per token row of a slot (16 mod 128)
  const int SLOTB = m * XROW;
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 0] = gtime();
#endif
  if (threadIdx.x == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NW); }
    mbar_init(excbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (CL > 1) cluster_arrive();  // matched by a wait before the first DSMEM access

  if (warp == NW) {
    // ---------------- x producer: TMA ring ----------------
    const uint32_t bytes_tok = XK * XCH;
    for (int si = 0; si < nslots; ++si) {
      const int st = si % NSLOT;
      if (si >= NSLOT) mbar_wait(&empty[st], ((si / NSLOT) - 1) & 1);
      if (lane == 0) mbar_expect_tx(&full[st], bytes_tok * m);
      __syncwarp();
      const size_t goff = (size_t)(c0 + si * XK) * XCH;
      for (int tok = lane; tok < m; tok += 32)
        tma_load_1d(xs + st * SLOTB + tok * XROW, reinterpret_cast<const uint8_t*>(p.x) + (size_t)tok * p.K * 2 + goff, bytes_tok, &full[st]);
#ifndef XWAITARRIVE
      if (si == XFIRST - 1 || (si == nslots - 1 && nslots < XFIRST)) { __syncwarp(); asm volatile("bar.arrive 2, %0;" :: "r"((NW + 1) * 32) : "memory"); }
#endif
    }
    if (CL > 1) { cluster_wait(); cluster_sync_all(); }
    return;
  }
  if (warp == NW + 1) {
    // ---------------- exception helper: exc_acc[row][tok] += x[tok][k] * delta ----------------
    for (int i = lane; i < ROWS * MT; i += 32) exc_acc[i] = 0.f;
    __syncwarp();
    if (!RAW) {
#ifdef XWAITARRIVE
      mbar_wait(&full[0], 0);
#endif
#ifdef TIMING
      if (lane == 0) p.tdbg[132 * 8 + bid * 4 + 0] = gtime();
#endif
      const int e0 = __ldg(&p.exc_off[bid]), e1 = __ldg(&p.exc_off[bid + 1]);
      const int ne = e1 - e0;
#ifdef TIMING
      if (lane == 0) p.tdbg[132 * 8 + bid * 4 + 1] = gtime() + 0 * ne;
#endif
      // lane = token; entries broadcast by shuffle; x loads batched 8 at a time; plain RMW (no races)
      const unsigned short* xb = reinterpret_cast<const unsigned short*>(p.x) + (size_t)lane * p.K;
      const bool act = lane < m;
      for (int base = 0; base < ne; base += 32) {
        int4 ent = make_int4(0, 0, 0, 0);
        if (base + lane < ne) ent = __ldg(&p.exc[e0 + base + lane]);
        const int cnt = min(32, ne - base);
        for (int j0 = 0; j0 < cnt; j0 += 8) {
          unsigned short xr[8];
          int rows[8];
          float ds[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const int j = j0 + u;
            const int jj = j < cnt ? j : 0;
            rows[u] = __shfl_sync(0xffffffffu, ent.x, jj);
            const int k = __shfl_sync(0xffffffffu, ent.y, jj);
            ds[u] = __int_as_float(__shfl_sync(0xffffffffu, ent.z, jj));
            xr[u] = 0;
            if (act && j < cnt) asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(xr[u]) : "l"(xb + k));
          }
#pragma unroll
          for (int u = 0; u < 8; ++u)
            if (act && j0 + u < cnt) exc_acc[rows[u] * MT + lane] += __uint_as_float(((uint32_t)xr[u]) << 16) * ds[u];
        }
      }
    }
    __syncwarp();
#ifdef TIMING
    if (lane == 0) p.tdbg[132 * 8 + bid * 4 + 3] = gtime();
#endif
    if (lane == 0) mbar_arrive(excbar);
    if (CL > 1) { cluster_wait(); cluster_sync_all(); }
    return;
  }

  // ---------------- compute warps ----------------
  const int wt = warp % WT, wk = warp / WT;
  const int g = lane >> 2, t = lane & 3;
  const int nq = nch / WK;                      // units for this warp
  const uint8_t* wbase = p.w + ((size_t)rg * CPR + c0 + wk) * (WT * UB) + wt * UB + lane * 16;
  auto uaddr = [&](int q) -> const uint8_t* { return wbase + (size_t)q * WK * (WT * UB); };
#ifdef XWAITARRIVE
  mbar_wait(&full[0], 0);  // x slot 0 has landed before the weight stream starts
#else
  asm volatile("bar.sync 2, %0;" :: "r"((NW + 1) * 32) : "memory");  // first x slots requested before the weights
#endif
  uint4 buf[D][NP];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < nq) {
      const uint8_t* u = uaddr(d);
#pragma unroll
      for (int pp = 0; pp < NP; ++pp) buf[d][pp] = ldg_stream(u + pp * 512);
    }
  float acc[MB][4];
#pragma unroll
  for (int i = 0; i < MB; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  constexpr int QPS = XK / WK;                  // units per slot per warp

  for (int q0 = 0; q0 < nq; q0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int q = q0 + d;
      if (q < nq) {
        const int si = q / QPS, jq = q - si * QPS;
        const int st = si % NSLOT;
        if (jq == 0) {
          mbar_wait(&full[st], (si / NSLOT) & 1);
#ifdef TIMING
          if (threadIdx.x == 0 && si == 0) p.tdbg[bid * 8 + 2] = gtime();
#endif
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
        if (q + D < nq) {
          const uint8_t* u = uaddr(q + D);
#pragma unroll
          for (int pp = 0; pp < NP; ++pp) buf[d][pp] = ldg_stream(u + pp * 512);
        }
        const uint8_t* xq = xs + st * SLOTB + (jq * WK + wk) * XCH;
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
          const int trow = (mb * 8 + g) < m ? (mb * 8 + g) : (m - 1);  // padded token rows alias a real row
          const uint8_t* xr = xq + trow * XROW;
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
        if (jq == QPS - 1) {
          __syncwarp();
          if (lane == 0) mbar_arrive(&empty[st]);
        }
      }
    }
  }
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 4] = gtime();
#endif
  // ---------------- reduction ----------------
  const float sc = p.scale;
  // exact corrections for out-of-window weights (computed by the helper warp) are added by wk == 0 warps
  mbar_wait(excbar, 0);
  if (!RAW && wk == 0) {
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mb][e] += exc_acc[(wt * 16 + g + (e >> 1) * 8) * MT + mb * 8 + 2 * t + (e & 1)];
  }
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 1] = gtime();
#endif
  if (WK > 1) {
    float* rw = red + (size_t)wk * ROWS * MT;
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
      for (int e = 0; e < 4; ++e) rw[(wt * 16 + g + (e >> 1) * 8) * MT + mb * 8 + 2 * t + (e & 1)] = acc[mb][e];
    bar_sync_id(1, NW * 32);
  }
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 3] = gtime();
#endif
  auto total = [&](int r, int tok) -> float {
    float v = 0.f;
#pragma unroll
    for (int k2 = 0; k2 < WK; ++k2) v += red[((size_t)k2 * ROWS + r) * MT + tok];
    return v;
  };
  if (CL == 1) {
    if (WK == 1) {
#pragma unroll
      for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int tok = mb * 8 + 2 * t + (e & 1);
          if (tok < m) store_out<OUTF32>(p.out, (size_t)tok * p.N + rg * ROWS + wt * 16 + g + (e >> 1) * 8, acc[mb][e] * sc);
        }
    } else {
      for (int idx = threadIdx.x; idx < ROWS * m; idx += NW * 32) {
        const int tok = idx / ROWS, r = idx - tok * ROWS;
        store_out<OUTF32>(p.out, (size_t)tok * p.N + rg * ROWS + r, total(r, tok) * sc);
      }
    }
  } else {
    cluster_wait();  // every CTA of the cluster is running (DSMEM targets alive)
    auto push = [&](int r, int tok, float v) {
      uint32_t ra;
      const uint32_t la = smem_u32(pbuf + ((size_t)rank * RO + r / CL) * MT + tok);
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"((uint32_t)(r % CL)));
      asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(ra), "f"(v) : "memory");
    };
    if (WK == 1) {
#pragma unroll
      for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int tok = mb * 8 + 2 * t + (e & 1);
          if (tok < m) push(wt * 16 + g + (e >> 1) * 8, tok, acc[mb][e]);
        }
    } else {
      for (int idx = threadIdx.x; idx < ROWS * m; idx += NW * 32) {
        const int r = idx / m, tok = idx - r * m;
        push(r, tok, total(r, tok));
      }
    }
#ifdef TIMING
    if (threadIdx.x == 0) p.tdbg[bid * 8 + 6] = gtime();
#endif
    cluster_sync_all();
#ifdef TIMING
    if (threadIdx.x == 0) p.tdbg[bid * 8 + 7] = gtime();
#endif
    for (int idx = threadIdx.x; idx < RO * m; idx += NW * 32) {
      const int tok = idx / RO, rr = idx - tok * RO;
      const int r = rr * CL + rank;
      if (r >= ROWS) continue;
      float v = 0.f;
#pragma unroll
      for (int c = 0; c < CL; ++c) v += pbuf[((size_t)c * RO + rr) * MT + tok];
      store_out<OUTF32>(p.out, (size_t)tok * p.N + rg * ROWS + r, v * sc);
    }
  }
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 5] = gtime();
#endif
}

template <int WT, int WK, int MB, int D, int CL, int XK, int NSLOT, bool RAW>
static size_t smem_bytes(int m) {
  constexpr int XCH = RAW ? 64 : 128, ROWS = WT * 16, RO = (ROWS + CL - 1) / CL, MT = MB * 8;
  size_t b = 256 + (size_t)ROWS * MT * 4 + (size_t)(WK > 1 ? WK : 0) * ROWS * MT * 4 + (CL > 1 ? (size_t)CL * RO * MT * 4 : 0);
  b = (b + 127) & ~(size_t)127;
  b += (size_t)NSLOT * m * (XK * XCH + 16) + 128;
  return b;
}

template <int WT, int WK, int MB, int D, int CL, int XK, int NSLOT, bool RAW>
static void launch_t(const Params& p, int grid) {
  const size_t smem = smem_bytes<WT, WK, MB, D, CL, XK, NSLOT, RAW>(p.m);
  TORCH_CHECK(smem <= 227 * 1024, "skinny: shared memory ", smem, " exceeds the limit");
  auto kern = skinny_kernel<WT, WK, MB, D, CL, XK, NSLOT, RAW>;
  static int cur = 0;
  if ((int)smem > cur) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cur = (int)smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3((WT * WK + 2) * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, p);
}

// mode: 0 = 12-bit compressed bf16 -> bf16, 1 = raw hi/lo router (x duplicated) -> fp32
void skinny(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor exc_off, torch::Tensor exc,
            int64_t N, int64_t CPR, double scale, int64_t mode, int64_t WT, int64_t WK, int64_t D, int64_t CL,
            int64_t XK, int64_t NSLOT, torch::Tensor tdbg) {
  Params p;
  p.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.out = out.data_ptr();
  p.exc_off = exc_off.data_ptr<int>();
  p.exc = reinterpret_cast<const int4*>(exc.data_ptr());
  p.m = (int)x.size(0);
  p.K = (int)x.size(1);
  p.N = (int)N;
  p.CPR = (int)CPR;
  p.scale = (float)scale;
  p.tdbg = reinterpret_cast<unsigned long long*>(tdbg.data_ptr());
  const int MB = (p.m + 7) / 8;
  const int g = (int)((N / (16 * WT)) * CL);
  DISPATCH_TABLE
  TORCH_CHECK(false, "skinny: no instantiation for WT=", WT, " WK=", WK, " D=", D, " CL=", CL, " XK=", XK, " NSLOT=", NSLOT, " MB=", MB, " mode=", mode);
}


__global__ void ldg_stream_k(const uint4* src, size_t n16, unsigned* out) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nt = (size_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (size_t i = tid; i < n16; i += nt * 4) {
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) { size_t j = i + u * nt; if (j < n16) v[u] = ldg_stream(src + j); else v[u] = make_uint4(0,0,0,0); }
#pragma unroll
    for (int u = 0; u < 4; ++u) acc ^= v[u].x;
  }
  if (acc == 0x12345u) out[0] = acc;
}
void stream(torch::Tensor src, torch::Tensor out, int64_t grid, int64_t threads) {
  ldg_stream_k<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), src.numel() / 16, (unsigned*)out.data_ptr());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("skinny", &skinny); m.def("stream", &stream); }
"""

_ext = {}


def _dispatch_src(insts):
    lines = []
    for (WT, WK, MB, D, CL, XK, NS, mode) in sorted(set(insts)):
        raw = "true" if mode == 1 else "false"
        lines.append(f"  if (WT == {WT} && WK == {WK} && MB == {MB} && D == {D} && CL == {CL} && XK == {XK} && NSLOT == {NS} && mode == {mode}) "
                     f"return launch_t<{WT}, {WK}, {MB}, {D}, {CL}, {XK}, {NS}, {raw}>(p, g);")
    return "\n".join(lines)


def _load_ext(insts):
    key = tuple(sorted(set(insts)))
    if key not in _ext:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        defs = os.environ.get("SKINNY_DEFS", "").split()
        src = CUDA_SRC.replace("DISPATCH_TABLE", _dispatch_src(insts))
        h = hashlib.sha1((src + " ".join(defs)).encode()).hexdigest()[:12]
        _ext[key] = load_inline(
            name=f"muse_proj_skinny_{h}",
            cpp_sources="",
            cuda_sources=src,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"] + defs,
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _ext[key]


def _frag_order(w, n_rows, kv, wt):
    """[N, Kv] -> [RG, CH, WT, 4(s), 8(g), 4(t), 2(qhi), 2(h), 2(qlo)]: mma.m16n8k16 A-fragment order.

    Row = rg*16*wt + tile*16 + h*8 + g, k = ch*64 + 16*t + 4*s + 2*qhi + qlo, so a lane's 16 k values
    per chunk are contiguous and its x (B operand) slice is a contiguous 32-byte run."""
    RG, CH = n_rows // (16 * wt), kv // 64
    v = w.reshape(RG, wt, 2, 8, CH, 4, 4, 2, 2)  # rg, tile, h, g, ch, t, s, qhi, qlo
    return v.permute(0, 4, 1, 6, 3, 5, 7, 2, 8)


def _pack_raw(w, wt):
    n_rows, kv = w.shape
    return _frag_order(w, n_rows, kv, wt).contiguous().view(torch.uint8).reshape(-1)


MAXE = 127  # exception entries per CTA (one smem slot holds the count)


def _pack_compressed(w, wt, cl):
    """Lossless 12-bit packing.  Weight w = sign|E|mantissa is stored as the byte sign|mantissa plus the
    4-bit code E - base, for the 16-exponent window [base, base+15] holding the most weights.  The kernel
    decodes exactly to sign|(112+code)|mantissa = w * 2^(112-base) and scales the fp32 result by
    2^(base-112).  Weights outside the window are stored as code 0 / byte 0 (decoding to 2^(base-127))
    and corrected exactly via a per-CTA list (row, k, delta) applied as out += x[k] * delta.

    Returns (packed bytes, exc_off, exc, scale) or None if the format does not fit."""
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
    RG, CH = n_rows // (16 * wt), K // 64
    smo = _frag_order(sm, n_rows, K, wt)  # rg, ch, tile, s, g, t, qhi, h, qlo
    smo = smo.reshape(RG, CH, wt, 2, 2, 8, 4, 2, 2, 2).permute(0, 1, 2, 3, 5, 6, 4, 7, 8, 9)
    smb = smo.to(torch.uint8).reshape(RG, CH, wt, 2, 512)
    nb = _frag_order(nib, n_rows, K, wt).permute(0, 1, 2, 4, 5, 3, 8, 6, 7)  # rg,ch,tile,g,t,s,qlo,qhi,h
    nb = nb.reshape(RG, CH, wt, 32, 4, 8).to(torch.int64)
    words = (nb << (torch.arange(8, device=w.device, dtype=torch.int64) * 4)).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    nbytes = words.contiguous().view(torch.uint8).reshape(RG, CH, wt, 1, 512)
    packed = torch.cat([smb, nbytes], dim=3).reshape(-1).contiguous()
    # exceptions grouped per CTA (rg, rank)
    rows, ks = (~inwin).nonzero(as_tuple=True)
    G = RG * cl
    if rows.numel():
        ch = ks // 64
        rank = torch.div(ch * cl, CH, rounding_mode="floor")
        # rank r owns chunks [r*CH/cl, (r+1)*CH/cl): fix boundary rounding exactly
        lo = torch.div(rank * CH, cl, rounding_mode="floor")
        rank = torch.where(ch < lo, rank - 1, rank)
        hi = torch.div((rank + 1) * CH, cl, rounding_mode="floor")
        rank = torch.where(ch >= hi, rank + 1, rank)
        cta = (rows // (16 * wt)) * cl + rank
        order = torch.argsort(cta, stable=True)
        rows, ks, cta = rows[order], ks[order], cta[order]
        counts = torch.bincount(cta, minlength=G)
        if int(counts.max()) > MAXE - 1:
            return None
        wt_ = w[rows, ks].double()
        delta = ((wt_ - 2.0 ** (base - 127)) * 2.0 ** (112 - base)).float()
        exc = torch.stack([(rows % (16 * wt)).to(torch.int32), ks.to(torch.int32), delta.view(torch.int32),
                           torch.zeros_like(ks, dtype=torch.int32)], dim=1).contiguous()
    else:
        counts = torch.zeros(G, dtype=torch.int64, device=w.device)
        exc = torch.zeros((1, 4), dtype=torch.int32, device=w.device)
    exc_off = torch.zeros(G + 1, dtype=torch.int32, device=w.device)
    exc_off[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return packed, exc_off, exc, 2.0 ** (base - 112)


def _cfg(wt, wk, d, cl, xk, ns):
    return {mb: (wt, wk, d, cl, xk, ns) for mb in (1, 2, 3, 4)}


# (layer, variant) -> {MB: (WT tiles per row group, WK k-split warps, D prefetch depth, CL cluster size,
#                         XK chunks per x slot, NSLOT x ring depth)}
CONFIGS = {
    ("qkv_gate", "full"): _cfg(3, 4, 4, 2, 4, 4),
    ("o", "full"): _cfg(4, 2, 4, 1, 4, 4),
    ("pooler", "full"): _cfg(4, 4, 4, 2, 4, 4),
    ("unpooler", "full"): _cfg(8, 2, 4, 2, 4, 4),
    ("unpooler", "slice"): _cfg(1, 8, 4, 2, 8, 4),
    ("router", "full"): _cfg(1, 8, 4, 4, 8, 4),
}


if os.environ.get("SKINNY_CFG_OVERRIDE"):  # local tuning only
    import ast
    for _k, _v in ast.literal_eval(os.environ["SKINNY_CFG_OVERRIDE"]).items():
        CONFIGS[_k] = _cfg(*_v)


def _instantiations(configs):
    out = []
    for (layer, _), per_mb in configs.items():
        mode = 1 if layer == "router" else 0
        for mb, (WT, WK, D, CL, XK, NS) in per_mb.items():
            out.append((WT, WK, mb, D, CL, XK, NS, mode))
    return out


def _layer_weight(weights, layer, variant, hi=None, lo=None):
    if layer == "router":
        return torch.stack([hi, lo], dim=2).reshape(256, -1).contiguous()  # interleaved [hi_k, lo_k]
    w = weights[layer].contiguous()
    return w[:1024].contiguous() if variant == "slice" else w


def build_packs(weights, configs, hi, lo, packs=None, dev=None):
    packs = {} if packs is None else packs
    for (layer, variant), per_mb in configs.items():
        wv = None
        for mb, (WT, WK, D, CL, XK, NS) in per_mb.items():
            key = (layer, variant, WT, CL)
            if key in packs:
                continue
            if wv is None:
                wv = _layer_weight(weights, layer, variant, hi, lo)
            cpr = wv.shape[1] // 64
            if layer == "router":
                packs[key] = ("rtr", _pack_raw(wv, WT), None, None, 1.0, wv.shape[0], cpr)
            else:
                res = _pack_compressed(wv, WT, CL)
                packs[key] = ("raw", None, None, None, 1.0, wv.shape[0], cpr) if res is None else ("cmp",) + tuple(res[:3]) + (res[3], wv.shape[0], cpr)
    return packs


def prepare(weights: Weights):
    dev = weights["qkv_gate"].device
    ext = _load_ext(_instantiations(CONFIGS))
    st = {"raw": {k: v.contiguous() for k, v in weights.items()}, "ext": ext, "configs": CONFIGS}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["hi"], st["lo"] = hi, lo
    st["router_hi_t"] = hi.t().contiguous()
    st["router_lo_t"] = lo.t().contiguous()
    st["packs"] = build_packs(weights, CONFIGS, hi, lo)
    st["dummy_off"] = torch.zeros(1024, dtype=torch.int32, device=dev)
    st["dummy_exc"] = torch.zeros((1, 4), dtype=torch.int32, device=dev)
    st["tdbg"] = torch.zeros(132 * 8 * 4 + 132 * 4, dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    return st


def run_skinny(st, x, layer, variant, cfg):
    WT, WK, D, CL, XK, NS = cfg
    kind, pk, eo, ex, sc, N, CPR = st["packs"][(layer, variant, WT, CL)]
    m = x.shape[0]
    if kind == "raw":
        return None
    assert (CPR // CL) % XK == 0 and CPR % CL == 0 and XK % WK == 0
    if kind == "rtr":
        out = torch.empty((m, N), dtype=torch.float32, device=x.device)
        st["ext"].skinny(pk, x, out, st["dummy_off"], st["dummy_exc"], N, CPR, 1.0, 1, WT, WK, D, CL, XK, NS, st["tdbg"])
    else:
        out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
        st["ext"].skinny(pk, x, out, eo, ex, N, CPR, sc, 0, WT, WK, D, CL, XK, NS, st["tdbg"])
    return out


def kernel_fn(data: Inputs, st) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    if m <= 32:
        variant = "slice" if (layer == "unpooler" and n == 1024) else "full"
        out = run_skinny(st, x, layer, variant, st["configs"][(layer, variant)][(m + 7) // 8])
        if out is not None:
            return out
    raw = st["raw"]
    if layer != "router":
        return F.linear(x, raw[layer][:n])
    if m <= 8:
        return F.linear(x.float(), raw["router"])
    acc = torch.mm(x, st["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, st["router_lo_t"], out_dtype=torch.float32)
    return acc

STREAM_SIZES = []
SWEEP = []
def add(layer, mbs, cfgs, var="x"):
    for c in cfgs:
        for mb in mbs:
            SWEEP.append((layer, var, mb, c))
# (WT, WK, D, CL, XK, NSLOT)
add("qkv_gate", [1, 2, 4], [(3, 4, 4, 2, 4, 4), (3, 4, 4, 2, 8, 4), (3, 4, 2, 2, 8, 4), (6, 2, 4, 4, 4, 4), (6, 2, 4, 4, 8, 4), (3, 4, 4, 2, 8, 6)])
add("qkv_gate", [1], [(3, 4, 4, 2, 8, 4), (6, 2, 4, 4, 8, 4)], var="m8")
add("pooler", [1, 2, 4], [(4, 4, 4, 2, 4, 4), (4, 4, 4, 2, 8, 4), (4, 4, 2, 2, 8, 4), (8, 2, 4, 4, 8, 4)])
add("pooler", [1], [(4, 4, 4, 2, 8, 4)], var="m8")
add("unpooler", [2, 4], [(8, 2, 4, 2, 4, 4), (8, 2, 4, 2, 8, 4), (4, 4, 4, 1, 8, 4), (8, 2, 2, 2, 8, 4)])
add("unpooler", [1], [(1, 8, 4, 2, 8, 4), (1, 8, 4, 2, 16, 2), (2, 4, 4, 2, 8, 4), (1, 8, 2, 2, 8, 4)])
add("unpooler", [1], [(1, 8, 4, 2, 8, 4), (2, 4, 4, 2, 8, 4)], var="m8")
add("o", [1, 2, 4], [(4, 2, 4, 1, 4, 4), (4, 2, 4, 1, 8, 2), (4, 4, 4, 1, 4, 4), (4, 4, 2, 1, 8, 2)])
add("o", [1], [(4, 2, 4, 1, 4, 4), (4, 4, 2, 1, 8, 2)], var="m8")
add("router", [1, 2, 4], [(1, 8, 4, 4, 8, 4), (1, 8, 4, 4, 16, 4), (1, 8, 2, 4, 16, 4), (2, 4, 4, 4, 8, 4)])
add("router", [1], [(1, 8, 4, 4, 8, 4), (1, 8, 4, 4, 16, 4)], var="m8")


# ======================= H200 diagnostic sweep (check runs only) =======================
_orig_prepare = prepare


def prepare(weights):
    import statistics, time
    os.environ["SKINNY_DEFS"] = "-DTIMING"
    t0 = time.time()
    allcfg = dict(CONFIGS)
    insts = _instantiations(CONFIGS)
    for (layer, variant, mb, cfg) in SWEEP:
        insts.append((cfg[0], cfg[1], mb, cfg[2], cfg[3], cfg[4], cfg[5], 1 if layer == "router" else 0))
    ext = _load_ext(insts)
    print(f"PROBE compile {time.time()-t0:.1f}s insts {len(set(insts))}", flush=True)
    st = _orig_prepare(weights)
    st["ext"] = ext
    dev = weights["qkv_gate"].device
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    junk = torch.zeros(1, device=dev)

    def graph(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def tmed(g, reps=15):
        ts = []
        for _ in range(reps):
            flush.zero_()
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(ts)

    fixed = tmed(graph(lambda: junk.add_(1)), 30)
    print(f"PROBE fixed {fixed:.2f}", flush=True)
    outbuf = torch.zeros(4, dtype=torch.int32, device=dev)
    for mbytes in STREAM_SIZES:
        src = torch.empty(int(mbytes * 1e6) // 16 * 16, dtype=torch.uint8, device=dev)
        for (grid_mult, thr) in ((1, 512), (1, 1024), (2, 512)):
            g = graph(lambda: ext.stream(src, outbuf, st["nsm"] * grid_mult, thr))
            t = tmed(g) - fixed
            print(f"STREAM {mbytes:5.1f}MB g{grid_mult} t{thr}: {t:6.2f}us {mbytes/t:5.2f}TB/s", flush=True)

    gen = torch.Generator(device=dev).manual_seed(1)
    names = ["xready", "loop_end", "end"]
    for (layer, variant, mb, cfg) in SWEEP:
        m = {1: 1, 2: 16, 3: 24, 4: 32}[mb] if variant != "m8" else 8
        var = "slice" if (layer == "unpooler" and m <= 8) else "full"
        K = weights[layer].shape[1]
        x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        try:
            build_packs(weights, {(layer, var): {mb: cfg}}, st["hi"], st["lo"], st["packs"])
        except Exception as e:
            print("SWEEP pack fail", layer, cfg, e); continue
        if st["packs"][(layer, var, cfg[0], cfg[3])][0] == "raw":
            print(f"SWEEP {layer} m{m} {cfg}: no compressed pack", flush=True); continue
        ref = F.linear(x.float(), weights[layer][: (1024 if var == "slice" else weights[layer].shape[0])].float())
        occn = -1
        try:
            g = graph(lambda: run_skinny(st, x, layer, var, cfg))
        except Exception as e:
            print(f"SWEEP {layer} m{m} {cfg}: FAIL {str(e)[:80]}", flush=True); torch.cuda.synchronize(); continue
        out = run_skinny(st, x, layer, var, cfg); torch.cuda.synchronize()
        err = float((out.float() - ref).norm() / ref.norm())
        t = tmed(g) - fixed
        flush.zero_(); st["tdbg"].zero_(); torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize()
        td = st["tdbg"][: 132 * 8].view(-1, 8)[:, :6].double()
        valid = td[:, 0] > 0
        td = td[valid]
        rel = (td - td[:, 0].min()) / 1000.0
        brk = " ".join(f"{nm} {rel[:, i].median():.2f}/{rel[:, i].max():.2f}" for nm, i in zip(names, (2, 4, 5)))
        print(f"SWEEP {layer:8s} m{m:<2d} {cfg}: {t:6.2f}us err {err:.1e} | {brk}", flush=True)
    os.environ["SKINNY_DEFS"] = ""
    print(f"PROBE total {time.time()-t0:.1f}s", flush=True)
    return st
