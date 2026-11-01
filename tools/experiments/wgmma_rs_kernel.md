# Parked experiment: wgmma (RS) compressed-weight kernel for m >= 16

Removed from submission.py (never instantiated in any submitted run). Findings are in the PR description:
correct results, but slow (weights stream too shallow with 1 warpgroup/SM; x re-read traffic; ptxas serializes
wgmma when A registers are redefined while a group is in flight).

## CUDA
```cpp
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


```

## Python glue
```python
def _wg_dispatch_src(insts):
    lines = []
    for (NR, NG, NT, D, CL, XK, NS, mode) in sorted(set(insts)):
        raw = "true" if mode == 1 else "false"
        lines.append(f"  if (NR == {NR} && NG == {NG} && NTOK == {NT} && D == {D} && CL == {CL} && XK == {XK} && NSLOT == {NS} && mode == {mode}) "
                     f"return wg_launch_t<{NR}, {NG}, {NT}, {D}, {CL}, {XK}, {NS}, {raw}>(p, g, map);")
    return "\n".join(lines)


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


    if not (layer == "unpooler" and n == 1024):
        for nt in WG_NTOK:
            if m <= nt:
                c = st["wg_configs"].get((layer, "full"), {}).get(nt)
                if c is not None and (m > 32 or st["configs"][(layer, "full")].get(_bucket(m)) is None):
                    out = run_wg(st, x, layer, "full", c)
                    if out is not None:
                        return out
                break

```
