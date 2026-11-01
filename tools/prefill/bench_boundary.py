import torch, triton, triton.language as tl, sys
sys.path.insert(0, "muse_prefill_layer")
import importlib.util
spec = importlib.util.spec_from_file_location("sub", "muse_prefill_layer/submission.py"); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
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

@triton.jit
def bnd2(a_ptr, s_ptr, alpha_ptr, beta_ptr, w_ptr, s1_ptr, h_ptr, N: tl.constexpr, CH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, CH)
    acc = tl.zeros([CH], tl.float32)
    for c in tl.static_range(N // CH):
        a = tl.load(a_ptr + row * N + c * CH + offs).to(tl.float32)
        acc += a * a
    ra = tl.rsqrt(tl.sum(acc, 0) / N + 1e-8)
    acc2 = tl.zeros([CH], tl.float32)
    for c in tl.static_range(N // CH):
        o = c * CH + offs
        a = tl.load(a_ptr + row * N + o).to(tl.float32)
        s = tl.load(s_ptr + row * N + o)
        s1 = tl.load(alpha_ptr + o) * s + tl.load(beta_ptr + o) * (a * ra)
        tl.store(s1_ptr + row * N + o, s1)
        acc2 += s1 * s1
    rs = tl.rsqrt(tl.sum(acc2, 0) / N + 1e-5)
    for c in tl.static_range(N // CH):
        o = c * CH + offs
        s1 = tl.load(s1_ptr + row * N + o)
        tl.store(h_ptr + row * N + o, (s1 * rs * tl.load(w_ptr + o)).to(tl.bfloat16))

T, N = 16384, 8192
a = torch.randn(T, N, device=dev).bfloat16(); s = torch.randn(T, N, device=dev)
al = torch.rand(N, device=dev); be = torch.rand(N, device=dev); w = torch.rand(N, device=dev)
s1 = torch.empty(T, N, device=dev); h = torch.empty(T, N, device=dev, dtype=torch.bfloat16)
by = T * N * (2 + 4 + 4 + 2)
for nw in [4, 8, 16, 32]:
    t = timeit(lambda: sub._attn_boundary_kernel[(T,)](a, s, al, be, w, s1, h, N=N, num_warps=nw))
    print(f"orig nw={nw}: {t:.1f} us {by / t / 1e3:.0f} GB/s")
ref_s1 = s1.clone(); ref_h = h.clone()
for nw, ch in [(8, 2048), (8, 4096), (16, 4096), (4, 1024)]:
    t = timeit(lambda: bnd2[(T,)](a, s, al, be, w, s1, h, N=N, CH=ch, num_warps=nw))
    print(f"chunked nw={nw} ch={ch}: {t:.1f} us {by / t / 1e3:.0f} GB/s  maxdiff {float((h.float()-ref_h.float()).abs().max()):.2e}")
# pure copy roofline
t = timeit(lambda: s1.copy_(s)); print(f"fp32 copy {t:.1f} us {T*N*8/t/1e3:.0f} GB/s")
