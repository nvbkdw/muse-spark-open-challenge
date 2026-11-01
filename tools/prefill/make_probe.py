"""Build a diagnostic check-run file from the submission: H200 microbenchmarks in prepare(), and during the
eager (non-captured) test calls, print the real per-expert counts and sweep humming configs on them.
usage: python tools/prefill/make_probe.py OUT.py"""
import sys
sub = open("muse_prefill_layer/submission.py").read()
probe = r'''

# ============================ diagnostic probe (check runs only) ============================
import time as _time
_PROBE_SEEN = set()


def _ev_time(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    ts = []
    flush = _PROBE_FLUSH
    for _ in range(reps):
        flush.fill_(1)
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


@triton.jit
def _bnd_chunk(a_ptr, s_ptr, alpha_ptr, beta_ptr, w_ptr, s1_ptr, h_ptr, N: tl.constexpr, CH: tl.constexpr):
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


_orig_prepare = prepare


def prepare(weights):
    global _PROBE_FLUSH
    st = _orig_prepare(weights)
    dev = weights["qkv_gate"].device
    _PROBE_FLUSH = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    for T in (3012, 16384):
        N = HIDDEN
        a = torch.randn(T, N, device=dev).bfloat16(); s = torch.randn(T, N, device=dev)
        s1 = torch.empty(T, N, device=dev); h = torch.empty(T, N, device=dev, dtype=torch.bfloat16)
        by = T * N * 12
        for nw in (4, 8, 16, 32):
            t = _ev_time(lambda: _attn_boundary_kernel[(T,)](a, s, st["a1"], st["b1"], st["w_pre_ffn"], s1, h, N=N, num_warps=nw))
            print(f"PROBE bnd T={T} orig nw={nw}: {t:.1f} us {by / t / 1e3:.0f} GB/s", flush=True)
        for nw, ch in ((8, 2048), (8, 4096), (16, 4096), (4, 2048)):
            t = _ev_time(lambda: _bnd_chunk[(T,)](a, s, st["a1"], st["b1"], st["w_pre_ffn"], s1, h, N=N, CH=ch, num_warps=nw))
            print(f"PROBE bnd T={T} chunk nw={nw} ch={ch}: {t:.1f} us {by / t / 1e3:.0f} GB/s", flush=True)
        t = _ev_time(lambda: s1.copy_(s))
        print(f"PROBE fp32 copy T={T}: {t:.1f} us {T * N * 8 / t / 1e3:.0f} GB/s", flush=True)
        del a, s, s1, h
    torch.cuda.empty_cache()
    return st


def _probe_moe(st, T, xg, h13, inter, y, cnt):
    srt = sorted(cnt.tolist(), reverse=True)
    print(f"PROBE counts T={T} routes={sum(srt)} distinct={sum(1 for c in srt if c)} {srt[:24]}", flush=True)
    if T in _PROBE_SEEN or T not in (753, 1506, 4096):
        return
    _PROBE_SEEN.add(T)
    t0 = _time.time()
    valid_m = 2 * T
    cnt_t = torch.tensor(srt, dtype=torch.int32, device=xg.device)  # placeholder, real one below
    res = {}
    for thr in (0, 16, 32, 64, 128):
        big = torch.where(cnt > thr, cnt, 0).int() if thr else cnt
        small = torch.where(cnt > thr, 0, cnt).int()
        for bmb in (128, 144, 160, 176):
            for bms in ((None,) if thr == 0 else (16, 32, 48)):
                if bms is not None and bms > max(thr, 16):
                    continue
                for (name, x, out) in (("w13", xg, h13), ("w2", inter, y)):
                    cb = _gemm_cfg(bmb, 256, 64)
                    cs = _gemm_cfg(bms, 256, 128) if bms else None
                    def fn():
                        _humming_run(st[name], x, out, big, cb, valid_m)
                        if cs is not None:
                            _humming_run(st[name], x, out, small, cs, valid_m)
                    try:
                        t = _ev_time(fn, reps=5)
                    except Exception as ex:
                        t = float("inf")
                    res.setdefault(name, []).append((t, f"thr{thr} big{bmb} small{bms}"))
    for name, lst in res.items():
        lst.sort()
        print(f"PROBE moe T={T} {name}: " + "; ".join(f"{t:.1f} {s}" for t, s in lst[:8]), flush=True)
    print(f"PROBE moe sweep T={T} took {_time.time() - t0:.1f}s", flush=True)


_orig_kernel_fn = kernel_fn


def kernel_fn(data, st):
    out = _orig_kernel_fn(data, st)
    if not torch.cuda.is_current_stream_capturing():
        T = data["config"]["tokens"]
        torch.cuda.synchronize()
        b = _buffers(st, T, data["attn_input"].device)
        _probe_moe(st, T, b["xg"], b["h13"], b["inter"], b["y"], st["cnt"].clone())
    return out
'''
open(sys.argv[1], "w").write(sub + probe)
