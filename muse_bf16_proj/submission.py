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
#include <cuda.h>
#include <stdint.h>

struct Params {
  const uint8_t* w;         // packed fragments [rg][chunk][tile][unit]
  const __nv_bfloat16* x;   // [m, K]
  void* out;                // [m, N]
  const int* exc_off;       // [grid + 1] exception CSR per CTA
  const int4* exc;          // {row in row group, k (actual), delta bits, 0}
  int m, N, K;              // K: actual x row length
  int CPR;                  // 64-wide (virtual) chunks per row
  int rotmul;               // chunk-order rotation multiplier (0 = none)
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
  float* pbuf = red + (WK > 1 ? WK : 0) * ROWS * (MT + 1);         // [CL][RO][MT] (CL > 1)
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
  const int rot = (int)(((unsigned)bid * (unsigned)p.rotmul) % (unsigned)nslots);  // optional rotated chunk order
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
      const size_t goff = (size_t)(c0 + ((si + rot) % nslots) * XK) * XCH;
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
  constexpr int QPS_ = XK / WK;
  const uint8_t* wbase = p.w + ((size_t)rg * CPR + c0 + wk) * (WT * UB) + wt * UB + lane * 16;
  auto uaddr = [&](int q) -> const uint8_t* {
    const int si = q / QPS_, jq = q - si * QPS_;
    return wbase + (size_t)(((si + rot) % nslots) * XK + jq * WK) * (WT * UB);
  };
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
    float* rw = red + (size_t)wk * ROWS * (MT + 1);
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
      for (int e = 0; e < 4; ++e) rw[(wt * 16 + g + (e >> 1) * 8) * (MT + 1) + mb * 8 + 2 * t + (e & 1)] = acc[mb][e];
    bar_sync_id(1, NW * 32);
  }
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 3] = gtime();
#endif
  auto total = [&](int r, int tok) -> float {
    float v = 0.f;
#pragma unroll
    for (int k2 = 0; k2 < WK; ++k2) v += red[((size_t)k2 * ROWS + r) * (MT + 1) + tok];
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
  size_t b = 256 + (size_t)ROWS * MT * 4 + (size_t)(WK > 1 ? WK : 0) * ROWS * (MT + 1) * 4 + (CL > 1 ? (size_t)CL * RO * MT * 4 : 0);
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

// =====================================================================================================
// wgmma variant (m >= 16): a warpgroup decodes 64 weight rows (4 tiles) into the A operand registers and
// multiplies with x read by the tensor cores from 128B-swizzled shared memory (B operand, K-major),
// filled by 2D TMA (zero-filled past m).  Router: hi/lo halves concatenated along K, x loaded twice.
// =====================================================================================================
struct WgParams {
  const uint8_t* w;
  void* out;
  const int* exc_off;
  const int4* exc;
  const __nv_bfloat16* x;
  int m, N, K, CPR;   // CPR: 64-wide chunks per weight row (virtual K / 64)
  float scale;
  unsigned long long* tdbg;
};

__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  // K-major, 128B swizzle: start (>>4), LBO = 1 (unused), SBO = 1024 B (8 rows x 128 B), swizzle mode 1 (128B)
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :: "r"(smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void fence_reg(uint32_t& r) { asm volatile("" : "+r"(r) :: "memory"); }
__device__ __forceinline__ void fence_reg(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
template <int NTOK> struct WgMma;
#define WG_DEF(NT, NREG)                                                                                       \
  template <> struct WgMma<NT> {                                                                               \
    static __device__ __forceinline__ void run(float* d, const uint32_t* a, uint64_t desc, int scale_d);       \
  };
WG_DEF(16, 8) WG_DEF(32, 16) WG_DEF(64, 32) WG_DEF(128, 64) WG_DEF(256, 128)

__device__ __forceinline__ void WgMma<16>::run(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
__device__ __forceinline__ void WgMma<32>::run(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
__device__ __forceinline__ void WgMma<64>::run(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
__device__ __forceinline__ void WgMma<128>::run(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}
__device__ __forceinline__ void WgMma<256>::run(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}


template <int NR, int NG, int NTOK, int D, int CL, int XK, int NSLOT, bool RAW>
__global__ void __launch_bounds__((NR * NG * 4 + 2) * 32, 1) wg_kernel(const WgParams p, const __grid_constant__ CUtensorMap xmap) {
  constexpr bool OUTF32 = RAW;
  constexpr int NW = NR * NG * 4;
  constexpr int NP = RAW ? 4 : 3;
  constexpr int UB = NP * 512;
  constexpr int ROWS = 64 * NR;
  constexpr int RO = ROWS / CL;
  constexpr int NACC = NTOK / 2;                // fp32 accumulators per thread (64 x NTOK per warpgroup)
  constexpr int CHB = NTOK * 128;               // smem bytes of one 64-wide x chunk (NTOK rows x 128 B)
  constexpr int WSB = XK * 4 * NR * UB;          // weight bytes per slot (contiguous [chunk][tile] block)
  constexpr int SLOTB = XK * CHB + WSB;
  static_assert(XK % NG == 0, "XK % NG");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* xs = smem;                                               // [NSLOT][XK][NTOK][128] swizzled
  uint64_t* full = reinterpret_cast<uint64_t*>(xs + NSLOT * SLOTB);
  uint64_t* empty = full + NSLOT;
  uint64_t* excbar = empty + NSLOT;
  float* exc_acc = reinterpret_cast<float*>(excbar + 2);            // [ROWS][NTOK]
  float* pbuf = exc_acc + ROWS * NTOK;                              // [CL][RO][NTOK] (CL > 1)
  float* red = reinterpret_cast<float*>(xs);                        // [NG][ROWS][NTOK] (reuses the x ring)

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int bid = blockIdx.x;
  const int rg = bid / CL, rank = bid % CL;
  const int CPR = p.CPR;
  const int c0 = (rank * CPR) / CL, c1 = ((rank + 1) * CPR) / CL;
  const int nch = c1 - c0;
  const int nslots = nch / XK;
  const int m = p.m;
  const int xch_per_row = RAW ? CPR / 2 : CPR;  // x chunks per token row
#ifdef NOROT
  const int rot = 0;
#else
  const int rot = (int)((bid * 7u) % (unsigned)nslots);
#endif
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 0] = gtime();
#endif
  if (threadIdx.x == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], NW); }
    mbar_init(excbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
  }
  __syncthreads();
  if (CL > 1) cluster_arrive();

  if (warp == NW) {
    // ---------------- x producer: 2D TMA (128B swizzle) per 64-wide chunk ----------------
    if (lane == 0) {
      for (int si = 0; si < nslots; ++si) {
        const int st = si % NSLOT;
        if (si >= NSLOT) mbar_wait(&empty[st], ((si / NSLOT) - 1) & 1);
        mbar_expect_tx(&full[st], SLOTB);
        const int psi = (si + rot) % nslots;  // rotated slot order: CTAs read different x chunks at a time
        tma_load_1d(xs + st * SLOTB + XK * CHB, p.w + ((size_t)rg * CPR + c0 + psi * XK) * (4 * NR * UB), WSB, &full[st]);
        for (int j = 0; j < XK; ++j) {
          int c = c0 + psi * XK + j;
          if (RAW && c >= xch_per_row) c -= xch_per_row;
          tma_load_2d(xs + st * SLOTB + j * CHB, &xmap, c * 64, 0, &full[st]);
        }
      }
    }
    __syncwarp();
    if (CL > 1) { cluster_wait(); cluster_sync_all(); }
    return;
  }
  if (warp == NW + 1) {
    // ---------------- exception helper (lane = token; 32 tokens per pass) ----------------
    for (int i = lane; i < ROWS * NTOK; i += 32) exc_acc[i] = 0.f;
    __syncwarp();
    if (!RAW) {
      const int e0 = __ldg(&p.exc_off[bid]), e1 = __ldg(&p.exc_off[bid + 1]);
      const int ne = e1 - e0;
      for (int tb = 0; tb < NTOK; tb += 32) {
        const int tok = tb + lane;
        const bool act = tok < m;
        const unsigned short* xb = reinterpret_cast<const unsigned short*>(p.x) + (size_t)(act ? tok : 0) * p.K;
        for (int base = 0; base < ne; base += 32) {
          int4 ent = make_int4(0, 0, 0, 0);
          if (base + lane < ne) ent = __ldg(&p.exc[e0 + base + lane]);
          const int cnt = min(32, ne - base);
          for (int j0 = 0; j0 < cnt; j0 += 8) {
            unsigned short xr[8]; int rows[8]; float ds[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              const int j = j0 + u, jj = j < cnt ? j : 0;
              rows[u] = __shfl_sync(0xffffffffu, ent.x, jj);
              const int k = __shfl_sync(0xffffffffu, ent.y, jj);
              ds[u] = __int_as_float(__shfl_sync(0xffffffffu, ent.z, jj));
              xr[u] = 0;
              if (act && j < cnt) asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(xr[u]) : "l"(xb + k));
            }
#pragma unroll
            for (int u = 0; u < 8; ++u)
              if (act && j0 + u < cnt) exc_acc[rows[u] * NTOK + tok] += __uint_as_float(((uint32_t)xr[u]) << 16) * ds[u];
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(excbar);
    if (CL > 1) { cluster_wait(); cluster_sync_all(); }
    return;
  }

  // ---------------- compute warpgroups ----------------
  const int wgi = warp >> 2, wr = wgi % NR, wg = wgi / NR, wt = (warp & 3) + 4 * wr;  // wt: 16-row tile in the CTA
  const int g = lane >> 2, t = lane & 3;
  const int nq = nch / NG;
  constexpr int QPS = XK / NG;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  uint32_t areg[2][16];
  const uint32_t xs_base = smem_u32(xs);
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 2] = gtime();
#endif

  for (int q = 0; q < nq; ++q) {
    {
      {
        const int si = q / QPS, jq = q - si * QPS;
        const int st = si % NSLOT;
        uint32_t* a = areg[q & 1];
        // the previous unit's wgmma group must be complete before its A registers are redefined
        // (overlapping would make ptxas serialize every wgmma); other warpgroups fill the gap.
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
        for (int i = 0; i < 16; ++i) fence_reg(a[i]);
        if (jq == 0) mbar_wait(&full[st], (si / NSLOT) & 1);
        const uint8_t* u = xs + st * SLOTB + XK * CHB + ((jq * NG + wg) * 4 * NR + wt) * UB + lane * 16;
        if (RAW) {
#pragma unroll
          for (int s = 0; s < 4; ++s) { const uint4 v = lds128(u + s * 512); a[4 * s] = v.x; a[4 * s + 1] = v.y; a[4 * s + 2] = v.z; a[4 * s + 3] = v.w; }
        } else {
          const uint4 p0 = lds128(u), p1 = lds128(u + 512), p2 = lds128(u + 1024);
          const uint32_t bw[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
          const uint32_t nw[4] = {p2.x, p2.y, p2.z, p2.w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            a[4 * s + 0] = dec(bw[2 * s], 0x1100, nw[s] << 7);
            a[4 * s + 1] = dec(bw[2 * s], 0x3322, nw[s] << 3);
            a[4 * s + 2] = dec(bw[2 * s + 1], 0x1100, nw[s] >> 1);
            a[4 * s + 3] = dec(bw[2 * s + 1], 0x3322, nw[s] >> 5);
          }
        }
        const uint32_t cb = xs_base + st * SLOTB + (jq * NG + wg) * CHB;
        uint64_t dsc[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) { dsc[s] = make_desc_sw128(cb + s * 32); asm volatile("" : "+l"(dsc[s])); }
#pragma unroll
        for (int i = 0; i < 16; ++i) fence_reg(a[i]);
#pragma unroll
        for (int i = 0; i < NACC; ++i) fence_reg(acc[i]);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#ifndef WGNOMMA
#pragma unroll
        for (int s = 0; s < 4; ++s) WgMma<NTOK>::run(acc, a + 4 * s, dsc[s], 1);
#else
        acc[0] += __uint_as_float(a[0] ^ a[5] ^ a[10] ^ a[15]);
#endif
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#pragma unroll
        for (int i = 0; i < NACC; ++i) fence_reg(acc[i]);
        if (jq == QPS - 1) {
          asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
          __syncwarp();
          if (lane == 0) mbar_arrive(&empty[st]);
        }
      }
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int i = 0; i < NACC; ++i) fence_reg(acc[i]);
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 4] = gtime();
#endif
  const float sc = p.scale;
  mbar_wait(excbar, 0);
  // acc layout (per n8 block j): acc[4j+0,1] = (row 16*wt+g, tok 8j+2t, +1); acc[4j+2,3] = (row +8, same toks)
  if (!RAW && wg == 0) {
#pragma unroll
    for (int j = 0; j < NTOK / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[4 * j + e] += exc_acc[(wt * 16 + g + (e >> 1) * 8) * NTOK + 8 * j + 2 * t + (e & 1)];
  }
  if (NG > 1) {
    // all warpgroups are done with the x ring (every slot was released after a full wait) -> reuse as red[]
    bar_sync_id(1, NW * 32);
    float* rw = red + (size_t)wg * ROWS * (NTOK + 1);
#pragma unroll
    for (int j = 0; j < NTOK / 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) rw[(wt * 16 + g + (e >> 1) * 8) * (NTOK + 1) + 8 * j + 2 * t + (e & 1)] = acc[4 * j + e];
    bar_sync_id(1, NW * 32);
  }
  auto total = [&](int r, int tok) -> float {
    float v = 0.f;
#pragma unroll
    for (int k2 = 0; k2 < NG; ++k2) v += red[((size_t)k2 * ROWS + r) * (NTOK + 1) + tok];
    return v;
  };
  if (CL == 1) {
    if (NG == 1) {
#pragma unroll
      for (int j = 0; j < NTOK / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int tok = 8 * j + 2 * t + (e & 1);
          if (tok < m) store_out<OUTF32>(p.out, (size_t)tok * p.N + rg * ROWS + wt * 16 + g + (e >> 1) * 8, acc[4 * j + e] * sc);
        }
    } else {
      for (int idx = threadIdx.x; idx < ROWS * m; idx += NW * 32) {
        const int tok = idx / ROWS, r = idx - tok * ROWS;
        store_out<OUTF32>(p.out, (size_t)tok * p.N + rg * ROWS + r, total(r, tok) * sc);
      }
    }
  } else {
    cluster_wait();
    auto push = [&](int r, int tok, float v) {
      uint32_t ra;
      const uint32_t la = smem_u32(pbuf + ((size_t)rank * RO + r / CL) * NTOK + tok);
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"((uint32_t)(r % CL)));
      asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(ra), "f"(v) : "memory");
    };
    if (NG == 1) {
#pragma unroll
      for (int j = 0; j < NTOK / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int tok = 8 * j + 2 * t + (e & 1);
          if (tok < m) push(wt * 16 + g + (e >> 1) * 8, tok, acc[4 * j + e]);
        }
    } else {
      for (int idx = threadIdx.x; idx < ROWS * m; idx += NW * 32) {
        const int r = idx / m, tok = idx - r * m;
        push(r, tok, total(r, tok));
      }
    }
    cluster_sync_all();
    for (int idx = threadIdx.x; idx < RO * m; idx += NW * 32) {
      const int tok = idx / RO, rr = idx - tok * RO;
      const int r = rr * CL + rank;
      float v = 0.f;
#pragma unroll
      for (int c = 0; c < CL; ++c) v += pbuf[((size_t)c * RO + rr) * NTOK + tok];
      store_out<OUTF32>(p.out, (size_t)tok * p.N + rg * ROWS + r, v * sc);
    }
  }
#ifdef TIMING
  if (threadIdx.x == 0) p.tdbg[bid * 8 + 5] = gtime();
#endif
}

typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                     const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                     CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static PFN_encodeTiled get_encode() {
  static PFN_encodeTiled fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* ptr = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &q);
    fn = reinterpret_cast<PFN_encodeTiled>(ptr);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled unavailable");
  }
  return fn;
}

template <int NR, int NG, int NTOK, int D, int CL, int XK, int NSLOT, bool RAW>
static void wg_launch_t(const WgParams& p, int grid, const CUtensorMap& map) {
  constexpr int ROWS = 64 * NR, RO = ROWS / CL, NW = NR * NG * 4;
  constexpr int UB_ = RAW ? 2048 : 1536;
  size_t ring = (size_t)NSLOT * (XK * NTOK * 128 + XK * 4 * NR * UB_);
  const size_t red = (size_t)NG * ROWS * (NTOK + 1) * 4;
  if (NG > 1 && red > ring) ring = red;
  const size_t smem = 1024 + ring + 64 + (size_t)ROWS * NTOK * 4 + (CL > 1 ? (size_t)CL * RO * NTOK * 4 : 0);
  TORCH_CHECK(smem <= 227 * 1024, "wg: shared memory ", smem, " exceeds the limit");
  auto kern = wg_kernel<NR, NG, NTOK, D, CL, XK, NSLOT, RAW>;
  static int cur = 0;
  if ((int)smem > cur) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); cur = (int)smem; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3((NW + 2) * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, p, map);
}

void wgrun(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor exc_off, torch::Tensor exc,
           int64_t N, int64_t CPR, double scale, int64_t mode, int64_t NR, int64_t NG, int64_t NTOK, int64_t D, int64_t CL,
           int64_t XK, int64_t NSLOT, torch::Tensor tdbg) {
  WgParams p;
  p.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  p.out = out.data_ptr();
  p.exc_off = exc_off.data_ptr<int>();
  p.exc = reinterpret_cast<const int4*>(exc.data_ptr());
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.m = (int)x.size(0);
  p.K = (int)x.size(1);
  p.N = (int)N;
  p.CPR = (int)CPR;
  p.scale = (float)scale;
  p.tdbg = reinterpret_cast<unsigned long long*>(tdbg.data_ptr());
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)p.K, (cuuint64_t)p.m};
  cuuint64_t gstride[1] = {(cuuint64_t)p.K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)NTOK};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p.x, gdim, gstride, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  const int g = (int)((N / (64 * NR)) * CL);
  WG_DISPATCH_TABLE
  TORCH_CHECK(false, "wgrun: no instantiation for NR=", NR, " NG=", NG, " NTOK=", NTOK, " D=", D, " CL=", CL, " XK=", XK, " NSLOT=", NSLOT, " mode=", mode);
}

// mode: 0 = 12-bit compressed bf16 -> bf16, 1 = raw hi/lo router (x duplicated) -> fp32
void skinny(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor exc_off, torch::Tensor exc,
            int64_t N, int64_t CPR, double scale, int64_t mode, int64_t WT, int64_t WK, int64_t D, int64_t CL,
            int64_t XK, int64_t NSLOT, torch::Tensor tdbg, int64_t rotmul) {
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
  p.rotmul = (int)rotmul;
  p.scale = (float)scale;
  p.tdbg = reinterpret_cast<unsigned long long*>(tdbg.data_ptr());
  const int MB = (p.m + 7) / 8;
  const int g = (int)((N / (16 * WT)) * CL);
  DISPATCH_TABLE
  TORCH_CHECK(false, "skinny: no instantiation for WT=", WT, " WK=", WK, " D=", D, " CL=", CL, " XK=", XK, " NSLOT=", NSLOT, " MB=", MB, " mode=", mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("skinny", &skinny); m.def("wgrun", &wgrun); }
"""

_ext = {}


def _dispatch_src(insts):
    lines = []
    for (WT, WK, MB, D, CL, XK, NS, mode) in sorted(set(insts)):
        raw = "true" if mode == 1 else "false"
        lines.append(f"  if (WT == {WT} && WK == {WK} && MB == {MB} && D == {D} && CL == {CL} && XK == {XK} && NSLOT == {NS} && mode == {mode}) "
                     f"return launch_t<{WT}, {WK}, {MB}, {D}, {CL}, {XK}, {NS}, {raw}>(p, g);")
    return "\n".join(lines)


def _wg_dispatch_src(insts):
    lines = []
    for (NR, NG, NT, D, CL, XK, NS, mode) in sorted(set(insts)):
        raw = "true" if mode == 1 else "false"
        lines.append(f"  if (NR == {NR} && NG == {NG} && NTOK == {NT} && D == {D} && CL == {CL} && XK == {XK} && NSLOT == {NS} && mode == {mode}) "
                     f"return wg_launch_t<{NR}, {NG}, {NT}, {D}, {CL}, {XK}, {NS}, {raw}>(p, g, map);")
    return "\n".join(lines)


def _load_ext(insts, wg_insts=()):
    key = (tuple(sorted(set(insts))), tuple(sorted(set(wg_insts))))
    if key not in _ext:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        defs = os.environ.get("SKINNY_DEFS", "").split()
        src = CUDA_SRC.replace("WG_DISPATCH_TABLE", _wg_dispatch_src(wg_insts)).replace("DISPATCH_TABLE", _dispatch_src(insts))
        h = hashlib.sha1((src + " ".join(defs)).encode()).hexdigest()[:12]
        _ext[key] = load_inline(
            name=f"muse_proj_skinny_{h}",
            cpp_sources="",
            cuda_sources=src,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"] + defs,
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            verbose=False,
        )
    return _ext[key]


def _frag_order(w, n_rows, kv, wt, natural=False):
    """[N, Kv] -> [RG, CH, WT, 4(s), 8(g), 4(t), 2(qhi), 2(h), 2(qlo)]: mma.m16n8k16 A-fragment order.

    Row = rg*16*wt + tile*16 + h*8 + g, k = ch*64 + 16*t + 4*s + 2*qhi + qlo, so a lane's 16 k values
    per chunk are contiguous and its x (B operand) slice is a contiguous 32-byte run."""
    RG, CH = n_rows // (16 * wt), kv // 64
    if natural:  # standard mma/wgmma k order: k = ch*64 + 16*s + 8*qhi + 2*t + qlo
        v = w.reshape(RG, wt, 2, 8, CH, 4, 2, 4, 2)  # rg, tile, h, g, ch, s, qhi, t, qlo
        return v.permute(0, 4, 1, 5, 3, 7, 6, 2, 8)
    v = w.reshape(RG, wt, 2, 8, CH, 4, 4, 2, 2)  # rg, tile, h, g, ch, t, s, qhi, qlo
    return v.permute(0, 4, 1, 6, 3, 5, 7, 2, 8)


def _pack_raw(w, wt, natural=False):
    n_rows, kv = w.shape
    return _frag_order(w, n_rows, kv, wt, natural).contiguous().view(torch.uint8).reshape(-1)


MAXE = 127  # exception entries per CTA (one smem slot holds the count)


def _pack_compressed(w, wt, cl, natural=False):
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
    smo = _frag_order(sm, n_rows, K, wt, natural)  # rg, ch, tile, s, g, t, qhi, h, qlo
    smo = smo.reshape(RG, CH, wt, 2, 2, 8, 4, 2, 2, 2).permute(0, 1, 2, 3, 5, 6, 4, 7, 8, 9)
    smb = smo.to(torch.uint8).reshape(RG, CH, wt, 2, 512)
    nb = _frag_order(nib, n_rows, K, wt, natural).permute(0, 1, 2, 4, 5, 3, 8, 6, 7)  # rg,ch,tile,g,t,s,qlo,qhi,h
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


# Per-shape kernel configuration, tuned on the judge GPU (H200) with diagnostic check runs.
# (layer, variant) -> {bucket: (WT tiles per row group, WK k-split warps, D prefetch depth, CL cluster size,
#                               XK chunks per x slot, NSLOT x ring depth)}, or None -> cuBLAS on the raw weights.
# bucket: "m1" for m == 1..7, then by MB = ceil(m / 8) (1 -> m 8, 2 -> m 9..16, 3, 4 -> m 17..32).
CONFIGS = {
    ("qkv_gate", "full"): {"m1": (3, 4, 4, 2, 16, 2), 1: (3, 4, 4, 2, 8, 4), 2: None, 3: None, 4: None},
    ("o", "full"): {"m1": (4, 4, 2, 1, 8, 2), 1: (4, 4, 2, 1, 8, 2), 2: (4, 4, 2, 1, 8, 2), 3: None, 4: None},
    ("pooler", "full"): {"m1": (4, 4, 2, 2, 8, 4), 1: (4, 4, 2, 2, 8, 4), 2: (4, 4, 2, 2, 8, 4),
                         3: (4, 4, 2, 2, 8, 4), 4: (4, 4, 2, 2, 8, 4)},
    ("unpooler", "full"): {2: (4, 4, 2, 1, 16, 2), 3: None, 4: None},
    ("unpooler", "slice"): {"m1": (1, 8, 4, 2, 16, 2), 1: (1, 8, 4, 2, 8, 4)},
    ("router", "full"): {"m1": (1, 8, 4, 4, 32, 2), 1: (1, 8, 4, 4, 16, 4), 2: (1, 8, 4, 4, 16, 4),
                         3: (1, 8, 4, 4, 16, 4), 4: (1, 8, 4, 4, 16, 4)},
}


# wgmma path: (layer, variant) -> {NTOK: (NR row warpgroups, NG k-split warpgroups, NTOK, D, CL, XK, NSLOT)}
WG_CONFIGS = {
    ("qkv_gate", "full"): {},
    ("o", "full"): {},
    ("pooler", "full"): {},
    ("unpooler", "full"): {},
    ("router", "full"): {},
}
WG_NTOK = (16, 32, 64, 128, 256)


def _wg_instantiations(cfgs):
    out = []
    for (layer, _), per in cfgs.items():
        for nt, c in per.items():
            if c is not None:
                out.append(tuple(c) + (1 if layer == "router" else 0,))
    return out


def _wg_weight(weights, layer, hi, lo):
    if layer == "router":
        return torch.cat([hi, lo], dim=1).contiguous()  # [256, 2K]: hi half then lo half (x used twice)
    return weights[layer].contiguous()


def build_wg_packs(weights, cfgs, hi, lo, packs):
    for (layer, variant), per in cfgs.items():
        for nt, c in per.items():
            if c is None:
                continue
            NR, NG, NT, D, CL, XK, NS = c
            key = ("wg", layer, variant, NR, CL)
            if key in packs:
                continue
            wv = _wg_weight(weights, layer, hi, lo)
            cpr = wv.shape[1] // 64
            if layer == "router":
                packs[key] = ("rtr", _pack_raw(wv, 4 * NR, natural=True), None, None, 1.0, wv.shape[0], cpr)
            else:
                res = _pack_compressed(wv, 4 * NR, CL, natural=True)
                packs[key] = ("raw", None, None, None, 1.0, wv.shape[0], cpr) if res is None else ("cmp",) + tuple(res[:3]) + (res[3], wv.shape[0], cpr)
    return packs


def run_wg(st, x, layer, variant, cfg):
    NR, NG, NT, D, CL, XK, NS = cfg
    kind, pk, eo, ex, sc, N, CPR = st["packs"][("wg", layer, variant, NR, CL)]
    m = x.shape[0]
    if kind == "raw" or m > NT:
        return None
    assert CPR % CL == 0 and (CPR // CL) % XK == 0 and XK % NG == 0
    if kind == "rtr":
        out = torch.empty((m, N), dtype=torch.float32, device=x.device)
        st["ext"].wgrun(pk, x, out, st["dummy_off"], st["dummy_exc"], N, CPR, 1.0, 1, NR, NG, NT, D, CL, XK, NS, st["tdbg"])
    else:
        out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
        st["ext"].wgrun(pk, x, out, eo, ex, N, CPR, sc, 0, NR, NG, NT, D, CL, XK, NS, st["tdbg"])
    return out


if os.environ.get("WG_CFG_OVERRIDE"):  # local tuning only: {(layer, ntok): cfg}
    import ast
    for (_l, _nt), _v in ast.literal_eval(os.environ["WG_CFG_OVERRIDE"]).items():
        WG_CONFIGS[(_l, "full")][_nt] = _v


ROTMUL = 7  # rotated per-CTA chunk order (A/B on the judge GPU: better or equal on every shape)


def _bucket(m):
    return "m1" if m < 8 else (m + 7) // 8


if os.environ.get("SKINNY_CFG_OVERRIDE"):  # local tuning only: {(layer, variant): cfg} for every bucket
    import ast
    for _k, _v in ast.literal_eval(os.environ["SKINNY_CFG_OVERRIDE"]).items():
        CONFIGS[_k] = {b: _v for b in ("m1", 1, 2, 3, 4)}


def _mb_of(bucket):
    return 1 if bucket == "m1" else bucket


def _instantiations(configs):
    out = []
    for (layer, _), per_b in configs.items():
        mode = 1 if layer == "router" else 0
        for b, cfg in per_b.items():
            if cfg is None:
                continue
            WT, WK, D, CL, XK, NS = cfg
            mbs = (1, 2, 3, 4) if b == "m1" else (b,)  # m1 config also serves m 2..7 (MB 1)
            for mb in ((1,) if b == "m1" else (b,)):
                out.append((WT, WK, mb, D, CL, XK, NS, mode))
    return out


def _layer_weight(weights, layer, variant, hi=None, lo=None):
    if layer == "router":
        return torch.stack([hi, lo], dim=2).reshape(256, -1).contiguous()  # interleaved [hi_k, lo_k]
    w = weights[layer].contiguous()
    return w[:1024].contiguous() if variant == "slice" else w


def build_packs(weights, configs, hi, lo, packs=None, dev=None):
    packs = {} if packs is None else packs
    for (layer, variant), per_b in configs.items():
        wv = None
        for b, cfg in per_b.items():
            if cfg is None:
                continue
            WT, WK, D, CL, XK, NS = cfg
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
    ext = _load_ext(_instantiations(CONFIGS), _wg_instantiations(WG_CONFIGS))
    st = {"raw": {k: v.contiguous() for k, v in weights.items()}, "ext": ext, "configs": CONFIGS, "wg_configs": WG_CONFIGS}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    st["hi"], st["lo"] = hi, lo
    st["router_hl_t"] = torch.cat([hi, lo], 0).t().contiguous()  # [K, 512]: one GEMM for hi and lo
    st["packs"] = build_packs(weights, CONFIGS, hi, lo)
    build_wg_packs(weights, WG_CONFIGS, hi, lo, st["packs"])
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
        st["ext"].skinny(pk, x, out, st["dummy_off"], st["dummy_exc"], N, CPR, 1.0, 1, WT, WK, D, CL, XK, NS, st["tdbg"], st.get("rotmul", ROTMUL))
    else:
        out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
        st["ext"].skinny(pk, x, out, eo, ex, N, CPR, sc, 0, WT, WK, D, CL, XK, NS, st["tdbg"], st.get("rotmul", ROTMUL))
    return out


def kernel_fn(data: Inputs, st) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    if m <= 32 or (layer == "router" and m <= 256):
        variant = "slice" if (layer == "unpooler" and n == 1024) else "full"
        cfg_ = st["configs"][(layer, variant)].get(_bucket(m))
        if cfg_ is not None:
            out = run_skinny(st, x, layer, variant, cfg_)
            if out is not None:
                return out
    if not (layer == "unpooler" and n == 1024):
        for nt in WG_NTOK:
            if m <= nt:
                c = st["wg_configs"].get((layer, "full"), {}).get(nt)
                if c is not None and (m > 32 or st["configs"][(layer, "full")].get(_bucket(m)) is None):
                    out = run_wg(st, x, layer, "full", c)
                    if out is not None:
                        return out
                break
    raw = st["raw"]
    if layer != "router":
        return F.linear(x, raw[layer][:n])
    if m <= 8:
        return F.linear(x.float(), raw["router"])
    o = torch.mm(x, st["router_hl_t"], out_dtype=torch.float32)  # BF16 hi/lo split, FP32 accumulation
    return o[:, :256] + o[:, 256:]
