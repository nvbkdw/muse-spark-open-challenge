import sys, time, torch
sys.path.insert(0, "../../../muse_prefill_layer"); sys.path.insert(0, ".")
import moe_fp4
from nvfp4 import random_weights, dequantize
dev = "cuda"
t0 = time.time(); moe_fp4._ext(); print("compile", time.time() - t0)
E, N, K = 32, 256, 4096
gen = torch.Generator(device=dev).manual_seed(1)
w, sf, gs = random_weights(N, K, generator=gen, device=dev, leading=(E,))
gs = gs * torch.exp(torch.randn(E, device=dev))
bank = moe_fp4.prepare_bank(w, sf, gs)
slab = 300
counts = torch.tensor([300, 1, 0, 17, 64, 255, 256, 257 % 300, 8, 9, 7] + [3] * 21, dtype=torch.int32, device=dev)
x = torch.randn(E * slab, K, device=dev, dtype=torch.bfloat16)
out = torch.zeros(E * slab, N, device=dev, dtype=torch.bfloat16)
moe_fp4.run(bank, x, counts, out, slab)
torch.cuda.synchronize()
for e in range(E):
    c = int(counts[e])
    if c == 0: continue
    wd = dequantize(w[e], sf[e], gs[e])
    ref = x[e * slab:e * slab + c].float() @ wd.T
    got = out[e * slab:e * slab + c].float()
    print(e, c, f"{float((got - ref).norm() / ref.norm()):.3e}")
