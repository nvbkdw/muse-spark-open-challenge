import sys, os, torch
sys.path.insert(0, "../../../muse_prefill_layer"); sys.path.insert(0, ".")
os.environ["MOE_FP4_BUILD"] = "/scratch-nvme/okc/moe_kernel/build_cfg"
import moe_fp4
from test_moe_fp4 import timeit, DISTS
from nvfp4 import random_weights
dev = "cuda"; E = 32; K = 4096
moe_fp4._ext()
if os.environ.get("COMPILE_ONLY"): sys.exit(0)
tag = " ".join(f"{k}={v}" for k, v in os.environ.items() if k.startswith("MOE_FP4_"))
for N in [8192, 4096]:
    gen = torch.Generator(device=dev).manual_seed(1)
    w, sf, gs = random_weights(N, K, generator=gen, device=dev, leading=(E,))
    bank = moe_fp4.prepare_bank(w, sf, gs)
    res = []
    for T in [753, 3012]:
        g = torch.Generator().manual_seed(T)
        perm = torch.randperm(32, generator=g)
        counts = torch.tensor(DISTS[T], dtype=torch.int32)[perm].to(dev)
        x = torch.randn(E * T, K, device=dev, dtype=torch.bfloat16)
        out = torch.empty(E * T, N, device=dev, dtype=torch.bfloat16)
        res.append(timeit(lambda: moe_fp4.run(bank, x, counts, out, T), 10))
    print(f"[{tag}] N={N} T753 {res[0]:.1f} T3012 {res[1]:.1f}", flush=True)
