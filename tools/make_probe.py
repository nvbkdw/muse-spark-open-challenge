"""Generate a diagnostic check-run file: submission.py + an H200 sweep executed inside prepare().

The sweep times kernel configs per shape like the judge (CUDA graph replay, cold L2, fixed
graph cost subtracted, median), and prints a compact table to the run log (log_tail).
usage: python tools/make_probe.py SWEEP_SPEC.py OUT.py
"""
import sys
sub = open("muse_bf16_proj/submission.py").read()
spec = open(sys.argv[1]).read()
stream_src = r'''
__global__ void ldg_stream_k(const uint4* src, size_t n16, unsigned* out) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nt = (size_t)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (size_t i = tid; i < n16; i += nt * 4) {
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) { size_t j = i + u * nt; if (j < n16) v[u] = ldg_stream(src + j); else v[u] = make_uint4(0,0,0,0); }
#pragma unroll
    for (int u = 0; u < 4; ++u) acc ^= v[u].x;
  }
  if (acc == 0x12345u) out[0] = acc;
}
void stream(torch::Tensor src, torch::Tensor out, int64_t grid, int64_t threads) {
  ldg_stream_k<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), src.numel() / 16, (unsigned*)out.data_ptr());
}
'''
sub = sub.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("skinny", &skinny); }',
                  stream_src + 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("skinny", &skinny); m.def("stream", &stream); }')
driver = r'''

# ======================= H200 diagnostic sweep (check runs only) =======================
_orig_prepare = prepare


def prepare(weights):
    import statistics, time
    os.environ["SKINNY_DEFS"] = "-DTIMING"
    t0 = time.time()
    allcfg = dict(CONFIGS)
    insts = _instantiations(CONFIGS)
    for (layer, variant, mb, cfg) in SWEEP:
        insts.append((cfg[0], cfg[1], mb, cfg[2], cfg[3], 1 if layer == "router" else 0))
    ext = _load_ext(insts)
    print(f"PROBE compile {time.time()-t0:.1f}s insts {len(set(insts))}", flush=True)
    st = _orig_prepare(weights)
    st["ext"] = ext
    dev = weights["qkv_gate"].device
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    junk = torch.zeros(1, device=dev)

    def graph(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def tmed(g, reps=15):
        ts = []
        for _ in range(reps):
            flush.zero_()
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(ts)

    fixed = tmed(graph(lambda: junk.add_(1)), 30)
    print(f"PROBE fixed {fixed:.2f}", flush=True)
    outbuf = torch.zeros(4, dtype=torch.int32, device=dev)
    for mbytes in STREAM_SIZES:
        src = torch.empty(int(mbytes * 1e6) // 16 * 16, dtype=torch.uint8, device=dev)
        for (grid_mult, thr) in ((1, 512), (1, 1024), (2, 512)):
            g = graph(lambda: ext.stream(src, outbuf, st["nsm"] * grid_mult, thr))
            t = tmed(g) - fixed
            print(f"STREAM {mbytes:5.1f}MB g{grid_mult} t{thr}: {t:6.2f}us {mbytes/t:5.2f}TB/s", flush=True)
    import reference_free_inputs as _ri  # noqa  (placeholder, replaced below)
'''
driver = driver.replace("    import reference_free_inputs as _ri  # noqa  (placeholder, replaced below)\n", r'''
    gen = torch.Generator(device=dev).manual_seed(1)
    names = ["xready", "loop_end", "flush_end"]
    for (layer, variant, mb, cfg) in SWEEP:
        m = {1: 1, 2: 16, 3: 24, 4: 32}[mb] if variant != "m8" else 8
        var = "slice" if (layer == "unpooler" and m <= 8) else "full"
        K = weights[layer].shape[1]
        x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
        build_packs(weights, {(layer, var): {mb: cfg}}, st["hi"], st["lo"], st["packs"])
        if st["packs"][(layer, var, cfg[0], cfg[1])][0] == "raw":
            print(f"SWEEP {layer} m{m} {cfg}: no compressed pack", flush=True); continue
        ref = F.linear(x.float(), weights[layer][: (1024 if var == "slice" else weights[layer].shape[0])].float())
        g = graph(lambda: run_skinny(st, x, layer, var, cfg))
        out = run_skinny(st, x, layer, var, cfg); torch.cuda.synchronize()
        err = float((out.float() - ref).norm() / ref.norm())
        t = tmed(g) - fixed
        flush.zero_(); st["tdbg"].zero_(); torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize()
        td = st["tdbg"].view(-1, 8)[:, :6].double()
        valid = td[:, 0] > 0
        td = td[valid]
        rel = (td - td[:, 0].min()) / 1000.0
        brk = " ".join(f"{nm} {rel[:, i].median():.2f}/{rel[:, i].max():.2f}" for nm, i in zip(names, (2, 4, 5)))
        print(f"SWEEP {layer:8s} m{m:<2d} W{cfg[0]} KB{cfg[1]} D{cfg[2]} DI{cfg[3]}: {t:6.2f}us err {err:.1e} | {brk}", flush=True)
    os.environ["SKINNY_DEFS"] = ""
    print(f"PROBE total {time.time()-t0:.1f}s", flush=True)
    return st
''')
open(sys.argv[2], "w").write(sub + "\n" + spec + driver)
