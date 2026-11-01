import sys, os, torch
sys.path.insert(0, "../../../muse_prefill_layer"); sys.path.insert(0, ".")
import moe_fp4
from test_moe_fp4 import timeit
from nvfp4 import random_weights
dev = "cuda"; E = 32
N = int(os.environ.get("N", "8192")); K = 4096
gen = torch.Generator(device=dev).manual_seed(1)
w, sf, gs = random_weights(N, K, generator=gen, device=dev, leading=(E,))
bank = moe_fp4.prepare_bank(w, sf, gs)
for spec in sys.argv[1:]:
    c, ne = map(int, spec.split("x"))  # count x num experts
    if c == 0: c, ne = 0, 0
    T = max(c, 16)
    counts = torch.tensor([c] * ne + [0] * (E - ne), dtype=torch.int32, device=dev)
    x = torch.randn(E * T, K, device=dev, dtype=torch.bfloat16)
    out = torch.empty(E * T, N, device=dev, dtype=torch.bfloat16)
    t = timeit(lambda: moe_fp4.run(bank, x, counts, out, T), 10)
    by = max(ne, 1) * N * K * 0.5625
    fl = 2 * c * ne * N * K
    print(f"{spec:10s} {t:8.1f} us  {by / t / 1e3:7.0f} GB/s  {fl / t / 1e6:6.0f} TFLOP/s", flush=True)
