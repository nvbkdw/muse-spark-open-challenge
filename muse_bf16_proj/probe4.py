# okc: problem=muse_bf16_proj
"""Diagnostic check run: HBM streaming vs CTA count / in-flight depth on the judge GPU; cluster occupancy."""
import os, statistics, torch
import torch.nn.functional as F

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
template <int U>
__global__ void stream_k(const uint4* __restrict__ src, size_t n16, unsigned* out, int per_cta16) {
  // CTA b streams a contiguous range; thread loads U x 16B per iteration (U in flight)
  const size_t b0 = (size_t)blockIdx.x * per_cta16;
  uint32_t acc = 0;
  for (size_t i = b0 + threadIdx.x; i < b0 + per_cta16 && i < n16; i += (size_t)blockDim.x * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      size_t j = i + (size_t)u * blockDim.x;
      if (j < b0 + per_cta16 && j < n16)
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(src + j));
      else v[u] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x;
  }
  if (acc == 0x12345u) out[0] = acc;
}
void stream(torch::Tensor src, torch::Tensor out, int64_t grid, int64_t threads, int64_t U) {
  size_t n16 = src.numel() / 16;
  int per = (int)((n16 + grid - 1) / grid);
  auto s = at::cuda::getCurrentCUDAStream();
  if (U == 2) stream_k<2><<<grid, threads, 0, s>>>((const uint4*)src.data_ptr(), n16, (unsigned*)out.data_ptr(), per);
  else if (U == 4) stream_k<4><<<grid, threads, 0, s>>>((const uint4*)src.data_ptr(), n16, (unsigned*)out.data_ptr(), per);
  else if (U == 8) stream_k<8><<<grid, threads, 0, s>>>((const uint4*)src.data_ptr(), n16, (unsigned*)out.data_ptr(), per);
  else stream_k<16><<<grid, threads, 0, s>>>((const uint4*)src.data_ptr(), n16, (unsigned*)out.data_ptr(), per);
}
__global__ void dummy(int* p) { extern __shared__ int s[]; if (p) p[0] = s[0]; }
int64_t occ(int64_t cl, int64_t smem, int64_t thr) {
  cudaFuncSetAttribute(dummy, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncSetAttribute(dummy, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(cl * 16); cfg.blockDim = dim3(thr); cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cl; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1; int n = -1; cudaOccupancyMaxActiveClusters(&n, dummy, &cfg); return n;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("stream", &stream); m.def("occ", &occ); }
"""


def prepare(weights):
    from torch.utils.cpp_extension import load_inline
    ext = load_inline("probe4_stream", "", SRC, extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"])
    dev = weights["qkv_gate"].device
    for cl in (2, 4, 8, 16):
        print("OCC cluster", cl, " ".join(f"smem{s//1024}K:{ext.occ(cl, s, 320)}" for s in (60 << 10, 120 << 10, 200 << 10)), flush=True)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    junk = torch.zeros(1, device=dev)
    out = torch.zeros(4, dtype=torch.int32, device=dev)

    def graph(fn):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def tmed(g, reps=15):
        ts = []
        for _ in range(reps):
            flush.zero_(); e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(ts)
    fixed = tmed(graph(lambda: junk.add_(1)), 30)
    for mb in (6.3, 12.6, 28.3, 50.3):
        src = torch.empty(int(mb * 1e6) // 16 * 16, dtype=torch.uint8, device=dev)
        res = []
        for grid in (64, 96, 128, 132):
            best = (1e9, None)
            for thr, U in ((256, 4), (256, 8), (256, 16), (512, 4), (512, 8), (1024, 4)):
                t = tmed(graph(lambda: ext.stream(src, out, grid, thr, U))) - fixed
                if t < best[0]: best = (t, (thr, U))
            res.append(f"G{grid}:{best[0]:.2f}us{best[1]}")
        print(f"STREAM {mb:5.1f}MB", " ".join(res), flush=True)
    return dict(weights)


def kernel_fn(data, st):
    x, layer = data["x"], data["config"]["layer"]
    if layer != "router":
        return F.linear(x, st[layer][: data["config"]["n"]])
    return F.linear(x.float(), st["router"])
