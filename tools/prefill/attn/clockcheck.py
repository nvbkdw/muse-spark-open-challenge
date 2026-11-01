import sys, torch, time, subprocess, threading
from common import *
import attn_impl as A
from sgl_kernel.flash_attn import flash_attn_with_kvcache
c = make_case(1, 32768, 16384, 0, 1)
L = 16384
f1 = lambda: flash_attn_with_kvcache(q=c["q"], k_cache=c["k_cache"], v_cache=c["v_cache"], page_table=c["page_table"],
    cache_seqlens=c["cache_seqlens"], cu_seqlens_q=c["cu_seqlens_q"], max_seqlen_q=L, softmax_scale=SCALE, causal=True)
f2 = lambda: A.custom_attention(c["q"], c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], L, SCALE, c["k_scale"], c["v_scale"])
for name, f in (("fa3", f1), ("mine", f2)):
    f(); torch.cuda.synchronize()
    t0 = time.time()
    while time.time() - t0 < 1.0:
        for _ in range(20): f()
        torch.cuda.synchronize()
    for _ in range(200): f()
    time.sleep(1.0)
    out = subprocess.run(["nvidia-smi", "-i", sys.argv[1], "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader"], capture_output=True, text=True).stdout
    torch.cuda.synchronize()
    print(name, out.strip())
