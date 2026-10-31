"""HBM streaming microbenchmark on the local GPU: 1-D TMA bulk pipeline vs LDG.128, cold L2."""
import os, statistics, torch
from torch.utils.cpp_extension import load_inline
SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
template<int STAGE, int NST>
__global__ void tma_stream(const uint8_t* src, size_t total, float* out, int evict) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint64_t* full = (uint64_t*)(sm + NST * STAGE);
  uint64_t* empty = full + NST;
  const int nw = blockDim.x / 32 - 1, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  size_t nst_total = total / STAGE;
  size_t s0 = nst_total * blockIdx.x / gridDim.x, s1 = nst_total * (blockIdx.x + 1) / gridDim.x;
  int n = (int)(s1 - s0);
  if (threadIdx.x == 0) {
    for (int i = 0; i < NST; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[i])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(&empty[i])), "r"(nw));
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  if (warp == nw) {
    if (lane == 0) for (int it = 0; it < n; ++it) {
      int st = it % NST;
      if (it >= NST) {
        uint32_t par = ((it / NST) - 1) & 1;
        asm volatile("{.reg .pred p; W1: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W1;}" :: "r"(sa(&empty[st])), "r"(par) : "memory");
      }
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(&full[st])), "r"(STAGE) : "memory");
      const uint8_t* g = src + (s0 + it) * STAGE;
      constexpr int PIECE = (STAGE % 16384 == 0) ? 16384 : 8192;
      for (int o = 0; o < STAGE; o += PIECE)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(sa(sm + st * STAGE + o)), "l"(g + o), "r"(PIECE), "r"(sa(&full[st])) : "memory");
    }
    return;
  }
  float acc = 0.f;
  for (int it = 0; it < n; ++it) {
    int st = it % NST; uint32_t par = (it / NST) & 1;
    asm volatile("{.reg .pred p; W2: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W2;}" :: "r"(sa(&full[st])), "r"(par) : "memory");
    const float4* b = (const float4*)(sm + st * STAGE);
    for (int i = threadIdx.x; i < STAGE / 16; i += nw * 32) { float4 v = b[i]; acc += v.x; }
    __syncwarp();
    if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[st])) : "memory");
  }
  if (acc == 1234.5f) out[0] = acc;
}
template<int UNROLL>
__global__ void ldg_stream(const uint4* src, size_t n16, float* out) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nt = (size_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (size_t i = tid; i < n16; i += nt * UNROLL) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) { size_t j = i + u * nt; if (j < n16) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(src + j)); else v[u] = make_uint4(0,0,0,0); }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x;
  }
  if (acc == 0x12345) out[0] = acc;
}
template<int STAGE, int NST>
void run_tma(torch::Tensor src, torch::Tensor out, int grid, int warps) {
  size_t smem = (size_t)NST * STAGE + 2 * NST * 8;
  auto k = tma_stream<STAGE, NST>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  k<<<grid, (warps + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>((const uint8_t*)src.data_ptr(), src.numel(), out.data_ptr<float>(), 0);
}
void tma(torch::Tensor src, torch::Tensor out, int grid, int warps, int stage, int nst) {
#define C(S, N) if (stage == S && nst == N) return run_tma<S, N>(src, out, grid, warps);
  C(8192, 4) C(8192, 8) C(8192, 12) C(8192, 16) C(16384, 4) C(16384, 6) C(16384, 8) C(16384, 12)
  C(24576, 4) C(24576, 6) C(24576, 8) C(32768, 3) C(32768, 4) C(32768, 6) C(49152, 4) C(4096, 16) C(4096, 32)
  TORCH_CHECK(false, "cfg");
}
void ldg(torch::Tensor src, torch::Tensor out, int grid, int threads, int unroll) {
  size_t n16 = src.numel() / 16;
  if (unroll == 4) ldg_stream<4><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), n16, out.data_ptr<float>());
  else ldg_stream<8><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), n16, out.data_ptr<float>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tma", &tma); m.def("ldg", &ldg); }
"""
ext = load_inline("stream_bench", "", SRC, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"])
nsm = torch.cuda.get_device_properties(0).multi_processor_count
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
out = torch.zeros(4, device="cuda")
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
junk = torch.zeros(1, device="cuda")
def graph_of(fn):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g
eg = graph_of(lambda: junk.add_(1))
def raw(g):
    ts = []
    for _ in range(15):
        flush.zero_(); e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)
FIXED = raw(eg)
print(f"fixed graph replay cost {FIXED:.2f} us")
def tm(fn, size):
    t = raw(graph_of(fn)) - FIXED; return t, size / t / 1e6
for size_mb in (int(x) for x in os.environ.get("SIZES", "28,50,400").split(",")):
    size = size_mb * 1024 * 1024 // 49152 * 49152
    src = torch.randint(0, 255, (size,), dtype=torch.uint8, device="cuda")
    print(f"--- {size/1e6:.1f} MB")
    for (th, un, gm) in [(256, 4, 1), (512, 4, 1), (512, 8, 1), (1024, 4, 1), (256, 8, 2), (512, 4, 2), (256, 4, 4)]:
        t, bw = tm(lambda: ext.ldg(src, out, nsm * gm, th, un), size); print(f"ldg thr{th} unroll{un} grid{gm}x: {t:6.2f} us {bw:5.2f} TB/s")
    for (st, nst, gm, w) in [(8192, 8, 1, 4), (8192, 16, 1, 4), (16384, 4, 1, 4), (16384, 8, 1, 4), (16384, 12, 1, 4), (24576, 4, 1, 4), (24576, 8, 1, 4), (32768, 4, 1, 4), (32768, 6, 1, 4), (49152, 4, 1, 4), (16384, 4, 2, 4), (24576, 4, 2, 4), (8192, 8, 2, 4), (4096, 16, 2, 4), (4096, 32, 1, 4)]:
        t, bw = tm(lambda: ext.tma(src, out, nsm * gm, w, st, nst), size); print(f"tma stage{st//1024}K x{nst} grid{gm}x: {t:6.2f} us {bw:5.2f} TB/s  (inflight/SM {st*nst*gm//1024}K)")
