"""Hypothesis: the 16-tile-variant monolithic kernel is slowed by ptxas wgmma serialization; a single-variant
(N=256) build should be much faster on large experts. Counts are multiples of 256 so only N=256 is needed."""
import os, sys, importlib, torch
sys.path.insert(0, "muse_prefill_layer"); sys.path.insert(0, "tools/prefill/moe_kernel")
os.environ["MOE_FP4_BUILD"] = "/scratch-nvme/okc/moe_kernel/sv_build"
which = sys.argv[1]
import moe_fp4 as M
if which == "single":
    M.NS = [256]
from nvfp4 import random_weights
dev = "cuda"
def timeit(fn, reps=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    fl = torch.empty(512 << 20, dtype=torch.uint8, device=dev); ts = []
    for _ in range(reps):
        fl.fill_(1); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
gen = torch.Generator(device=dev).manual_seed(1)
n, k = 8192, 4096
w, sf, gs = random_weights(n, k, generator=gen, device=dev, leading=(32,)); gs = gs * torch.ones(32, device=dev)
bank = M.prepare_bank(w, sf, gs)
for counts in ([512, 512, 256, 256] + [0] * 28, [1024, 1024, 512, 256, 256, 256] + [0] * 26, [256] * 8 + [0] * 24):
    T = max(counts)
    cnt = torch.tensor(counts, dtype=torch.int32, device=dev)
    x = torch.randn(32 * T, k, device=dev, dtype=torch.bfloat16)
    out = torch.empty(32 * T, n, device=dev, dtype=torch.bfloat16)
    t = timeit(lambda: M.run(bank, x, cnt, out, T))
    R = sum(counts)
    # correctness spot check expert 0
    from nvfp4 import dequantize
    ref = x[:counts[0]].float() @ dequantize(w[0], sf[0], gs.reshape(-1)[0]).T
    err = float((out[:counts[0]].float() - ref).norm() / ref.norm())
    print(f"{which}: counts={counts[:6]} routes={R}: {t:.1f} us {2 * R * n * k / t / 1e6:.0f} TFLOP/s relerr {err:.2e}", flush=True)
