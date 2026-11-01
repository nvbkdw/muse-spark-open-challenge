"""Grouped NVFP4-weight x BF16-activation MoE GEMM for Hopper (sm_90a), masked-slab layout.

Swap-AB: Y^T[n, tok] = W[n, k] * X^T[k, tok].  The dequantized weight is the wgmma A operand
(registers, m64 x k16 per warpgroup), the tokens are the wgmma N dimension (16..256, chosen per
expert on-device), X tiles arrive in shared memory through TMA (128B swizzle).

Weights are repacked losslessly (prepare_bank): 4 bits per weight, e4m3 block scales, fp32
global scale.  Each thread's A fragment for one k16 step is one uint32 whose bits are placed so
that 4 rotations + 5 LOP3 produce four bf16x2 registers holding e2m1 * 2^-126 exactly
(sign at bit 15, exponent/mantissa bits at 8..6), which are multiplied (bf16x2) by the block scale
converted to bf16 as sf * 2^118.  The epilogue multiplies by gs * 2^8 in fp32.

Public API:
  prepare_bank(w, sf, gs, fuse_gate_up=False) -> dict
  run(bank, x, counts, out, slab)                     Y = X @ W^T per expert
  run_gate_up_silu(bank, x, counts, out, slab)        out = silu(X@Wg^T) * (X@Wu^T), bank prepared
                                                      with fuse_gate_up=True
"""
from __future__ import annotations

import os

import torch

BM = 128          # weight rows per CTA tile (2 consumer warpgroups x 64)
BK = 64           # k per pipeline stage
KSTEP_BYTES = BM * BK // 2 + BM * BK // 16  # 4096 nibble bytes + 512 scale bytes
NS = list(range(16, 257, 16))  # token-tile sizes (wgmma N) instantiated

# Bit layout of one uint32 (8 nibbles, one k16 step of one thread's A fragment).  Output reg i
# (bf16x2: sign at bit 15/31, e1 e0 m at bits 8..6 / 24..22) is extracted as
#   a0 = q & M
#   a1 = (q << 6) & M
#   a2 = ((q << 10) | (q >> 22)) & M
#   a3 = ((q << 4) & M_A) | ((q >> 4) & M_B)          M_A = {7,8,15,23,24,31}, M_B = {6,22}
# = 7 ALU (LOP3/SHF) + 3 IMAD.SHL per 8 weights (found by exhaustive search; no 4-shift tiling exists).
_M = [6, 7, 8, 15, 22, 23, 24, 31]
_M_LO = [6, 7, 8, 15]  # bit for nibble bits 0(m),1(e0),2(e1),3(sign) of the low bf16
_M_A = {7, 8, 15, 23, 24, 31}


def _src_pos(ri, p):
    if ri == 0:
        return p
    if ri == 1:
        return p - 6
    if ri == 2:
        return p - 10 if p >= 10 else p + 22
    return p - 4 if p in _M_A else p + 4


def _bit_positions():
    """pos[ri][pair][b] = source bit in the packed uint32 of nibble bit b of element (ri, pair)."""
    pos = []
    for ri in range(4):
        pr = []
        for pair in range(2):
            pr.append([_src_pos(ri, _M_LO[b] + 16 * pair) for b in range(4)])
        pos.append(pr)
    flat = sorted(p for a in pos for b in a for p in b)
    assert flat == list(range(32)), flat
    return pos


def _gen_wgmma():
    out = []
    out.append("template<int N> __device__ __forceinline__ void wgmma_rs(float* d, const uint32_t* a, uint64_t desc);")
    for n in NS:
        nreg = n // 2
        douts = ", ".join(f"%{i}" for i in range(nreg))
        a = ", ".join(f"%{nreg + i}" for i in range(4))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
        out.append(f"""template<> __device__ __forceinline__ void wgmma_rs<{n}>(float* d, const uint32_t* a, uint64_t desc) {{
  asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{douts}}}, {{{a}}}, %{nreg + 4}, 1, 1, 1, 0;\\n"
    : {cons} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}}""")
    return "\n".join(out)


def _gen_dispatch():
    lines = []
    for n in NS:
        lines.append(f"      case {n // 16}: consume<{n}, FUSED>(p, it, sm, priv, fullb, emptyb, stage, phase, wg, wp, lane); break;")
    return "\n".join(lines)


CUDA_SRC = r"""
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

#define STAGES __STAGES__
#define STAGE_BYTES 37888
#define WSTAGE 4608
#define NTHREADS 384

struct Params {
  const uint8_t* wp;
  const int* counts;
  const float* gs;
  __nv_bfloat16* out;
  int slab, NB, KS, out_ld, E;
};

struct Item { int e, nb, c0, n, rows; };
// m[i]: 3D view [K/64][rows][64] of X with box {64, 16(i+1), kps(16(i+1))}: one TMA per stage
struct Maps { CUtensorMap m[16]; };

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt));
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  while (!ok) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  }
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template<int K> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(K) : "memory"); }
__device__ __forceinline__ void fence_u32(uint32_t& r) { asm volatile("" : "+r"(r) :: "memory"); }
__device__ __forceinline__ void fence_u64(uint64_t& r) { asm volatile("" : "+l"(r) :: "memory"); }
__device__ __forceinline__ void fence_f32(float& r) { asm volatile("" : "+f"(r) :: "memory"); }

__WGMMA__

// bf16x2 multiply by a broadcast half of s (folded into HMUL2 .H0_H0 / .H1_H1 by ptxas)
__device__ __forceinline__ uint32_t bmul_lo(uint32_t a, uint32_t s) {
  uint32_t d;
  asm("{\n .reg .b16 l, h;\n .reg .b32 t;\n mov.b32 {l, h}, %2;\n mov.b32 t, {l, l};\n mul.rn.bf16x2 %0, %1, t;\n}\n" : "=r"(d) : "r"(a), "r"(s));
  return d;
}
__device__ __forceinline__ uint32_t bmul_hi(uint32_t a, uint32_t s) {
  uint32_t d;
  asm("{\n .reg .b16 l, h;\n .reg .b32 t;\n mov.b32 {l, h}, %2;\n mov.b32 t, {h, h};\n mul.rn.bf16x2 %0, %1, t;\n}\n" : "=r"(d) : "r"(a), "r"(s));
  return d;
}
// two e4m3 scales (selected 16-bit half of v) -> bf16x2 (sf * 2^118), exact incl. e4m3 subnormals
template<int HI>
__device__ __forceinline__ uint32_t scale_pair(uint32_t v) {
  uint32_t h;
  if (HI) asm("{\n .reg .b16 l, hh;\n mov.b32 {l, hh}, %1;\n cvt.rn.f16x2.e4m3x2 %0, hh;\n}\n" : "=r"(h) : "r"(v));
  else asm("{\n .reg .b16 l, hh;\n mov.b32 {l, hh}, %1;\n cvt.rn.f16x2.e4m3x2 %0, l;\n}\n" : "=r"(h) : "r"(v));
  return (h >> 3) + 0x73007300u;
}

// q: packed nibbles for one k16 step; S: bf16x2 scales (row R0 low half, row R1 high half)
__device__ __forceinline__ void dequant_k16(uint32_t q, uint32_t S, uint32_t* a) {
  const uint32_t M = 0x81C081C0u, MA = 0x81808180u, MB = 0x00400040u;
  uint32_t r0 = q & M;
  uint32_t r1 = (q << 6) & M;
  uint32_t r2 = ((q << 10) | (q >> 22)) & M;
  uint32_t r3 = ((q << 4) & MA) | ((q >> 4) & MB);
  a[0] = bmul_lo(r0, S);
  a[1] = bmul_hi(r1, S);
  a[2] = bmul_lo(r2, S);
  a[3] = bmul_hi(r3, S);
}

__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62);
}

struct Sched {
  int order[32];
  int cnt[32];
  int n[32];
  int nch[32];
  int cum[33];
};

__device__ __forceinline__ Item decode(const Sched& s, int i, int NB) {
  int r = 0;
  while (s.cum[r + 1] <= i) ++r;
  Item it;
  it.e = s.order[r];
  int local = i - s.cum[r];
  int nch = s.nch[it.e];
  it.nb = local / nch;
  int ch = local - it.nb * nch;
  it.n = s.n[it.e];
  it.c0 = ch * it.n;
  it.rows = min(it.n, s.cnt[it.e] - it.c0);
  return it;
}

// Small token tiles (N <= SMALLN): each consumer thread streams its own weight fragment (16 B) and
// scales (8 B) with cp.async into a private smem ring, PD k-steps ahead (no barriers needed; the
// TMA engine is reserved for the activation tiles).  Large tiles: weights + X per stage via TMA.
#define SMALLN 128
#define MAXCHUNK __MAXCHUNK__
#define PD __PD__
__host__ __device__ constexpr int kps_for(int n) {
  return n > SMALLN ? 1 : (n <= 16 ? 16 : (n <= 32 ? 8 : (n <= 64 ? 4 : 2)));
}
__host__ __device__ constexpr int wblk_for(int n) { return n > SMALLN ? 5120 : 0; }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template<int K> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(K) : "memory"); }

template<int N, int FUSED>
__device__ __forceinline__ void consume(const Params& p, const Item& it, uint32_t sm, uint32_t priv,
                                        uint32_t fullb, uint32_t emptyb, int& stage, uint32_t& phase,
                                        int wg, int wp, int lane) {
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  constexpr int KPS = kps_for(N);
  constexpr int WBLK = wblk_for(N);
  const int tid = wp * 32 + lane;
  const uint32_t wofs = (wg * 128 + tid) * 16;
  const uint32_t sofs = 4096 + (wg * 32 + wp * 8 + (lane >> 2)) * 8;
  int rel = stage;
  const int KS = p.KS;
  if constexpr (N <= SMALLN) {
    const uint8_t* gsrc = p.wp + (size_t)(it.e * p.NB + it.nb) * KS * WSTAGE;
    const uint32_t pq = priv + (wg * 128 + tid) * 16;
    const uint32_t ps = priv + PD * 4096 + (wg * 128 + tid) * 8;
#pragma unroll
    for (int d = 0; d < PD; ++d) {
      cp_async16(pq + d * 4096, gsrc + d * WSTAGE + wofs);
      cp_async8(ps + d * 2048, gsrc + d * WSTAGE + sofs);
      cp_commit();
    }
    uint32_t A[2][16];
    for (int ks = 0; ks < KS; ks += 2) {
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int k = ks + u;
        const int kk = k & (KPS - 1);
        if (kk == 0) mbar_wait(fullb + stage * 8, phase);
        cp_wait<PD - 1>();
        const int slot = k & (PD - 1);
        uint32_t q[4], sc[2];
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(pq + slot * 4096));
        asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(sc[0]), "=r"(sc[1]) : "r"(ps + slot * 2048));
#pragma unroll
        for (int j = 0; j < 4; ++j)
          dequant_k16(q[j], (j & 1) ? scale_pair<1>(sc[j >> 1]) : scale_pair<0>(sc[j >> 1]), &A[u][4 * j]);
        if (k + PD < KS) {
          cp_async16(pq + slot * 4096, gsrc + (size_t)(k + PD) * WSTAGE + wofs);
          cp_async8(ps + slot * 2048, gsrc + (size_t)(k + PD) * WSTAGE + sofs);
        }
        cp_commit();
        const uint64_t desc = make_desc(sm + stage * STAGE_BYTES + kk * (N * 128));
        wgmma_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j) wgmma_rs<N>(acc, &A[u][4 * j], desc + 2 * j);
        wgmma_commit();
        wgmma_wait<1>();
#pragma unroll
        for (int i = 0; i < 16; ++i) fence_u32(A[u ^ 1][i]);
        if (kk == 0) {
          if (k >= 1 && lane == 0) mbar_arrive(emptyb + rel * 8);
        }
        if (kk == KPS - 1) {
          rel = stage;
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
      }
    }
    cp_wait<0>();
  } else {
    // one commit group per k16, two k16 fragments in flight (register-lean, for large N)
    uint32_t A[2][4];
    for (int ks = 0; ks < KS; ++ks) {
      mbar_wait(fullb + stage * 8, phase);
      const uint32_t wbase = sm + stage * STAGE_BYTES;
      uint32_t q[4], sc[2];
      asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(wbase + wofs));
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(sc[0]), "=r"(sc[1]) : "r"(wbase + sofs));
      const uint64_t desc = make_desc(wbase + WBLK);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        dequant_k16(q[j], (j & 1) ? scale_pair<1>(sc[j >> 1]) : scale_pair<0>(sc[j >> 1]), A[j & 1]);
        wgmma_fence();
        wgmma_rs<N>(acc, A[j & 1], desc + 2 * j);
        wgmma_commit();
        wgmma_wait<1>();
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_u32(A[(j & 1) ^ 1][i]);
        if (j == 0) {
          if (ks >= 1 && lane == 0) mbar_arrive(emptyb + rel * 8);
        }
      }
      rel = stage;
      if (++stage == STAGES) { stage = 0; phase ^= 1; }
    }
  }
  wgmma_wait<0>();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) fence_f32(acc[i]);
  if (lane == 0) mbar_arrive(emptyb + rel * 8);

  // epilogue
  const float sc = __ldg(p.gs + it.e) * 256.0f;
  const int g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)it.e * p.slab + it.c0;
  if (FUSED) {
    const int feat = it.nb * 64 + wg * 32 + wp * 8 + g;
#pragma unroll
    for (int j = 0; j < N / 8; ++j) {
#pragma unroll
      for (int hc = 0; hc < 2; ++hc) {
        const int col = 8 * j + 2 * t + hc;
        if (col < it.rows) {
          float gt = acc[4 * j + hc] * sc;
          float up = acc[4 * j + 2 + hc] * sc;
          float y = gt / (1.0f + __expf(-gt)) * up;
          p.out[(row0 + col) * p.out_ld + feat] = __float2bfloat16_rn(y);
        }
      }
    }
  } else {
    const int r = it.nb * 128 + wg * 64 + wp * 16 + g;
#pragma unroll
    for (int j = 0; j < N / 8; ++j) {
#pragma unroll
      for (int hc = 0; hc < 2; ++hc) {
        const int col = 8 * j + 2 * t + hc;
        if (col < it.rows) {
          __nv_bfloat16* o = p.out + (row0 + col) * p.out_ld + r;
          o[0] = __float2bfloat16_rn(acc[4 * j + hc] * sc);
          o[8] = __float2bfloat16_rn(acc[4 * j + 2 + hc] * sc);
        }
      }
    }
  }
}

template<int FUSED>
__global__ void __launch_bounds__(NTHREADS, 1)
moe_fp4_kernel(const __grid_constant__ Maps maps, const Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const uint32_t sm = smem_u32(smem);
  const uint32_t fullb = sm + STAGES * STAGE_BYTES;
  const uint32_t emptyb = fullb + STAGES * 8;
  Sched& s = *reinterpret_cast<Sched*>(smem + STAGES * STAGE_BYTES + STAGES * 16);
  const uint32_t priv = sm + STAGES * STAGE_BYTES + 1024;  // PD * 256 * 24 bytes

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp == 8) {
    // schedule: experts sorted by tile size desc, then count desc
    const int e = lane;
    int c = (e < p.E) ? p.counts[e] : 0;
    c = max(0, min(c, p.slab));
    int nch0 = (c + MAXCHUNK - 1) / MAXCHUNK;
    int n = nch0 ? ((((c + nch0 - 1) / nch0) + 15) & ~15) : 0;
    int nch = n ? (c + n - 1) / n : 0;
    long long key = ((long long)n << 40) | ((long long)c << 8) | (31 - e);
    int rank = 0;
    for (int o = 0; o < 32; ++o) {
      long long ko = __shfl_sync(0xffffffffu, key, o);
      rank += (ko > key);
    }
    s.order[rank] = e;
    s.cnt[e] = c;
    s.n[e] = n;
    s.nch[e] = nch;
    __syncwarp();
    int items = s.nch[s.order[lane]] * p.NB;
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_up_sync(0xffffffffu, items, o);
      if (lane >= o) items += v;
    }
    s.cum[lane + 1] = items;
    if (lane == 0) {
      s.cum[0] = 0;
      for (int i = 0; i < STAGES; ++i) {
        mbar_init(fullb + i * 8, 1);
        mbar_init(emptyb + i * 8, 8);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  __syncthreads();
  const int total = s.cum[32];
  int stage = 0;
  uint32_t phase = 0;

  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (warp == 8 && lane == 0) {
      for (int i = blockIdx.x; i < total; i += gridDim.x) {
        const Item it = decode(s, i, p.NB);
        if (it.rows <= 0) continue;
        const uint8_t* wsrc = p.wp + (size_t)(it.e * p.NB + it.nb) * p.KS * WSTAGE;
        const int row0 = it.e * p.slab + it.c0;
        const int kps = kps_for(it.n);
        const uint32_t wblk = wblk_for(it.n);
        const bool wtma = it.n > SMALLN;
        const uint32_t bytes = kps * ((wtma ? WSTAGE : 0) + it.n * 128);
        for (int ks = 0; ks < p.KS; ks += kps) {
          mbar_wait(emptyb + stage * 8, phase ^ 1);
          const uint32_t fb = fullb + stage * 8;
          const uint32_t sb = sm + stage * STAGE_BYTES;
          mbar_expect_tx(fb, bytes);
          if (wtma) bulk_g2s(sb, wsrc + (size_t)ks * WSTAGE, kps * WSTAGE, fb);
          tma3d(sb + wblk, &maps.m[(it.n >> 4) - 1], 0, row0, ks, fb);
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
    const int wg = warp >> 2, wp = warp & 3;
    for (int i = blockIdx.x; i < total; i += gridDim.x) {
      const Item it = decode(s, i, p.NB);
      if (it.rows <= 0) continue;
      switch (it.n >> 4) {
__DISPATCH__
        default: break;
      }
    }
  }
}

static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&fn, 12000, cudaEnableDefault, &q);
  }
  return fn;
}

static int make_map(CUtensorMap* m, void* ptr, uint64_t rows, uint64_t K, uint32_t box_rows, uint32_t box_k) {
  auto fn = get_encode();
  if (!fn) return -1;
  CUresult r;
  if (box_k == 0) {
    cuuint64_t dims[2] = {K, rows};
    cuuint64_t strides[1] = {K * 2};
    cuuint32_t box[2] = {64, box_rows};
    cuuint32_t es[2] = {1, 1};
    r = fn(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
           CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  } else {
    cuuint64_t dims[3] = {64, rows, K / 64};
    cuuint64_t strides[2] = {K * 2, 128};
    cuuint32_t box[3] = {64, box_rows, box_k};
    cuuint32_t es[3] = {1, 1, 1};
    r = fn(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
           CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  return (int)r;
}

static int smem_bytes() { return STAGES * STAGE_BYTES + 1024 + PD * 256 * 24 + 1024; }

int moe_fp4_launch(const void* wp, void* x, int64_t rows, int64_t K, const int* counts, int E, const float* gs,
                   void* out, int64_t out_ld, int64_t slab, int64_t NB, int64_t fused, int64_t grid, cudaStream_t stream) {
  Maps maps;
  int r = 0;
  for (int i = 0; i < 16; ++i) {
    r = make_map(&maps.m[i], x, rows, K, 16 * (i + 1), kps_for(16 * (i + 1)));
    if (r) return 1000 + r;
  }
  Params p;
  p.wp = (const uint8_t*)wp;
  p.counts = counts;
  p.gs = gs;
  p.out = (__nv_bfloat16*)out;
  p.slab = (int)slab;
  p.NB = (int)NB;
  p.KS = (int)(K / 64);
  p.out_ld = (int)out_ld;
  p.E = E;
  const int smem = smem_bytes();
  if (fused) {
    static bool init = false;
    if (!init) { cudaFuncSetAttribute(moe_fp4_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
    moe_fp4_kernel<1><<<grid, NTHREADS, smem, stream>>>(maps, p);
  } else {
    static bool init = false;
    if (!init) { cudaFuncSetAttribute(moe_fp4_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
    moe_fp4_kernel<0><<<grid, NTHREADS, smem, stream>>>(maps, p);
  }
  return (int)cudaGetLastError();
}
"""

CPP_SRC = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
int moe_fp4_launch(const void* wp, void* x, int64_t rows, int64_t K, const int* counts, int E, const float* gs,
                   void* out, int64_t out_ld, int64_t slab, int64_t NB, int64_t fused, int64_t grid, cudaStream_t stream);
void moe_fp4_run(torch::Tensor wp, torch::Tensor x, torch::Tensor counts, torch::Tensor gs, torch::Tensor out,
                 int64_t slab, int64_t NB, int64_t fused, int64_t grid) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16, "x must be contiguous bf16");
  TORCH_CHECK(out.stride(1) == 1 && out.scalar_type() == at::kBFloat16, "out must be row-major bf16");
  TORCH_CHECK(counts.scalar_type() == at::kInt && counts.numel() <= 32, "counts int32 [<=32]");
  TORCH_CHECK(x.size(1) % 128 == 0, "K % 128");
  int r = moe_fp4_launch(wp.data_ptr(), x.data_ptr(), x.size(0), x.size(1), counts.data_ptr<int>(), (int)counts.numel(),
                         gs.data_ptr<float>(), out.data_ptr(), out.stride(0), slab, NB, fused, grid,
                         at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "moe_fp4_launch failed: ", r);
}
"""

STAGES = int(os.environ.get("MOE_FP4_STAGES", "5"))
_EXT = None


def _source():
    return (CUDA_SRC.replace("__WGMMA__", _gen_wgmma())
            .replace("__DISPATCH__", _gen_dispatch())
            .replace("__STAGES__", str(STAGES))
            .replace("__PD__", os.environ.get("MOE_FP4_PD", "4"))
            .replace("__MAXCHUNK__", os.environ.get("MOE_FP4_MAXCHUNK", "256")))


_CUDA_FLAGS = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20", "--use_fast_math", "-lineinfo"]


def _build_manual(name):
    """Fallback when ninja is missing: nvcc + g++ by hand, then import the pybind module."""
    import hashlib
    import importlib.util
    import subprocess
    import sysconfig
    from torch.utils import cpp_extension as ce
    src = _source()
    h = hashlib.sha1((src + CPP_SRC + " ".join(_CUDA_FLAGS)).encode()).hexdigest()[:12]
    bdir = os.path.join(os.environ.get("MOE_FP4_BUILD", "/tmp/moe_fp4_build"), f"{name}_{h}")
    os.makedirs(bdir, exist_ok=True)
    so = os.path.join(bdir, f"{name}.so")
    if not os.path.exists(so):
        cu, cpp = os.path.join(bdir, "k.cu"), os.path.join(bdir, "m.cpp")
        open(cu, "w").write(src)
        open(cpp, "w").write(CPP_SRC + f"\nPYBIND11_MODULE({name}, m) {{ m.def(\"moe_fp4_run\", &moe_fp4_run); }}\n")
        inc = sum([["-I", p] for p in ce.include_paths("cuda")], []) + ["-I", sysconfig.get_paths()["include"]]
        abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
        nvcc = os.path.join(ce.CUDA_HOME, "bin", "nvcc")
        verbose = ["-Xptxas=-v"] if os.environ.get("MOE_FP4_VERBOSE") else []
        p1 = subprocess.Popen([nvcc, "-c", cu, "-o", cu + ".o", "-Xcompiler", "-fPIC", abi] + inc + _CUDA_FLAGS + verbose,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        subprocess.run(["g++", "-c", cpp, "-o", cpp + ".o", "-fPIC", "-O2", "-std=c++20", abi,
                        f"-DTORCH_EXTENSION_NAME={name}"] + inc, check=True)
        o1 = p1.communicate()[0]
        if os.environ.get("MOE_FP4_VERBOSE"):
            print(o1)
        if p1.returncode:
            raise RuntimeError(o1)
        libs = sum([["-L", p] for p in ce.library_paths("cuda")], [])
        subprocess.run(["g++", "-shared", cu + ".o", cpp + ".o", "-o", so] + libs +
                       ["-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10_cuda", "-ltorch_cuda", "-lcudart"],
                       check=True)
    spec = importlib.util.spec_from_file_location(name, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ext():
    global _EXT
    if _EXT is None:
        import shutil
        from torch.utils.cpp_extension import load_inline
        name = f"moe_fp4_s{STAGES}_v1"
        # torch adds -gencode for TORCH_CUDA_ARCH_LIST / the detected arch; plain sm_90 cannot
        # assemble wgmma/setmaxnreg, so pin the arch list to 9.0a for this build.
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        if shutil.which("ninja") is None:
            _EXT = _build_manual(name)
        else:
            _EXT = load_inline(name=name, cpp_sources=CPP_SRC, cuda_sources=_source(), functions=["moe_fp4_run"],
                               extra_cuda_cflags=_CUDA_FLAGS)
    return _EXT


# ------------------------------------------------------------------------------------------
# lossless repack
# ------------------------------------------------------------------------------------------
_POS = None


def _pos_table(device):
    global _POS
    if _POS is None or _POS.device != torch.device(device):
        pos = _bit_positions()
        # order of the 8 elements in the word: index = ri * 2 + pair, ri = 2 * hi + half
        t = torch.tensor([[pos[ri][pair][b] for b in range(4)] for ri in range(4) for pair in range(2)],
                         dtype=torch.int64, device=device)
        _POS = t
    return _POS


def _fused_perm(n_rows):
    """physical row -> logical row for gate/up interleave (gate rows 0..H-1, up rows H..2H-1)."""
    H = n_rows // 2
    p = torch.arange(n_rows)
    nb, w = p // 128, p % 128
    wg, w = w // 64, w % 64
    wp, w = w // 16, w % 16
    half, g = w // 8, w % 8
    feat = nb * 64 + wg * 32 + wp * 8 + g
    return torch.where(half == 0, feat, H + feat)


def prepare_bank(w, sf, gs, fuse_gate_up=False):
    """w uint8 [E, N, K/2], sf e4m3 [E, N, K/16], gs fp32 [E] -> repacked bank (lossless)."""
    E, N, K2 = w.shape
    K = 2 * K2
    assert N % BM == 0 and K % (2 * BK) == 0
    NB, KS = N // BM, K // BK
    dev = w.device
    pos = _pos_table(dev)
    shifts = torch.ones_like(pos) << pos  # [8, 4]
    perm = _fused_perm(N).to(dev) if fuse_gate_up else None
    out = torch.empty(E, NB, KS, KSTEP_BYTES, dtype=torch.uint8, device=dev)
    bits = torch.arange(4, device=dev, dtype=torch.uint8)
    for e in range(E):
        we = w[e]
        se = sf[e].view(torch.uint8)
        if perm is not None:
            we = we[perm]
            se = se[perm]
        codes = torch.stack((we & 0xF, we >> 4), dim=-1).reshape(N, K)  # [N, K] uint8
        sval = se.view(torch.float8_e4m3fn).float()
        zero = (sval == 0)
        neg = (se & 0x80) != 0
        cb = codes.view(N, K // 16, 16)
        cb = torch.where(zero.unsqueeze(-1), torch.zeros_like(cb), cb)
        cb = torch.where((neg & ~zero).unsqueeze(-1), cb ^ 8, cb)
        codes = cb.reshape(N, K)
        se = torch.where(zero, torch.zeros_like(se), se & 0x7F)
        # rows [NB, wg2, wp4, half2, g8], k [KS, j4, hi2, t4, pair2]
        c = codes.view(NB, 2, 4, 2, 8, KS, 4, 2, 4, 2)
        # -> [NB, KS, wg, wp, g, t, j, hi, half, pair]
        c = c.permute(0, 5, 1, 2, 4, 8, 6, 7, 3, 9).reshape(NB, KS, 2, 4, 8, 4, 4, 8)
        b = ((c.unsqueeze(-1) >> bits) & 1).to(torch.int64)  # [..., 8, 4]
        word = (b * shifts).sum(dim=(-1, -2))  # int64 [NB, KS, 2, 4, 8, 4, 4]
        word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
        wbytes = word.reshape(NB, KS, -1).view(torch.uint8)  # 4096 bytes
        s = se.view(NB, 2, 4, 2, 8, KS, 4).permute(0, 5, 1, 2, 4, 6, 3).reshape(NB, KS, 512)
        out[e, :, :, :4096] = wbytes
        out[e, :, :, 4096:] = s
    return {"wp": out, "gs": gs.float().contiguous(), "N": N, "K": K, "NB": NB, "fused": bool(fuse_gate_up)}


_NUM_SMS = {}


def _num_sms(dev):
    if dev not in _NUM_SMS:
        _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NUM_SMS[dev]


def run(bank, x, counts, out, slab, grid=None):
    """out[e*slab + i, :] = x[e*slab + i, :] @ W_e^T for i < counts[e]."""
    assert not bank["fused"]
    if grid is None:
        grid = _num_sms(x.device.index or 0)
    _ext().moe_fp4_run(bank["wp"], x, counts, bank["gs"], out, slab, bank["NB"], 0, grid)
    return out


def run_gate_up_silu(bank, x, counts, out, slab, grid=None):
    """out[e*slab+i, :H] = silu(x @ Wg^T) * (x @ Wu^T); bank prepared with fuse_gate_up=True."""
    assert bank["fused"]
    if grid is None:
        grid = _num_sms(x.device.index or 0)
    _ext().moe_fp4_run(bank["wp"], x, counts, bank["gs"], out, slab, bank["NB"], 1, grid)
    return out
