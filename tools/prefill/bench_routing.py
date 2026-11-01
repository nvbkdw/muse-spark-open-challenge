import torch, triton, triton.language as tl, importlib.util
spec = importlib.util.spec_from_file_location("sub", "muse_prefill_layer/submission.py"); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
dev = "cuda"
def timeit(fn, reps=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]

@triton.jit
def routing2(logit_ptr, bias_ptr, local_start_ptr, ids_ptr, w_ptr, rowmap_ptr, cnt_ptr, slab, T,
             LOGIT_STRIDE: tl.constexpr, HILO: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    t64 = t.to(tl.int64)
    e = tl.arange(0, 256)
    x = tl.load(logit_ptr + t64[:, None] * LOGIT_STRIDE + e[None, :], mask=tm[:, None], other=0.0)
    if HILO:
        x += tl.load(logit_ptr + t64[:, None] * LOGIT_STRIDE + 256 + e[None, :], mask=tm[:, None], other=0.0)
    sc = tl.sigmoid(x * 0.1767766952966369)
    b = sc + tl.load(bias_ptr + e)[None, :]
    j = tl.arange(0, 8)
    ids = tl.zeros([BT, 8], dtype=tl.int32)
    ws = tl.zeros([BT, 8], dtype=tl.float32)
    for k in tl.static_range(8):
        m = tl.max(b, 1)
        idx = tl.min(tl.where(b == m[:, None], e[None, :], 256), 1)
        hit = e[None, :] == idx[:, None]
        wk = tl.sum(tl.where(hit, sc, 0.0), 1)
        ids = tl.where(j[None, :] == k, idx[:, None], ids)
        ws = tl.where(j[None, :] == k, wk[:, None], ws)
        b = tl.where(hit, float("-inf"), b)
    ws = ws / (tl.sum(ws, 1)[:, None] + 1e-15)
    o = t64[:, None] * 8 + j[None, :]
    m2 = tm[:, None]
    tl.store(ids_ptr + o, ids, mask=m2)
    tl.store(w_ptr + o, ws, mask=m2)
    ls = tl.load(local_start_ptr)
    le = (ids - ls + 256) % 256
    is_local = (le < 32) & m2
    slot = tl.atomic_add(cnt_ptr + le, tl.full([BT, 8], 1, tl.int32), mask=is_local)
    tl.store(rowmap_ptr + o, tl.where(is_local, le * slab + slot, -1), mask=m2)

for T in [753, 3012, 16384]:
    logits = torch.randn(T, 512, device=dev)
    bias = torch.randn(256, device=dev) * 0.05
    ls = torch.tensor([0], dtype=torch.int32, device=dev)
    outs = []
    for variant in ["old", "new"]:
        ids = torch.empty(T, 8, dtype=torch.int32, device=dev); tw = torch.empty(T, 8, device=dev)
        rm = torch.empty(T, 8, dtype=torch.int32, device=dev); cnt = torch.zeros(32, dtype=torch.int32, device=dev)
        if variant == "old":
            fn = lambda: (cnt.zero_(), sub._routing_kernel[(T,)](logits, bias, ls, ids, tw, rm, cnt, T, LOGIT_STRIDE=512, HILO=True, num_warps=2))
            print(T, "old", f"{timeit(fn):.1f} us")
        else:
            for bt, nw in [(8, 4), (16, 4), (32, 8), (16, 8)]:
                fn = lambda: (cnt.zero_(), routing2[(triton.cdiv(T, bt),)](logits, bias, ls, ids, tw, rm, cnt, T, T, LOGIT_STRIDE=512, HILO=True, BT=bt, num_warps=nw))
                print(T, "new", bt, nw, f"{timeit(fn):.1f} us")
        cnt.zero_(); fn(); torch.cuda.synchronize()
        outs.append((ids.clone(), tw.clone(), cnt.clone()))
    print("  ids equal", bool((outs[0][0] == outs[1][0]).all()), "w maxdiff", float((outs[0][1] - outs[1][1]).abs().max()), "cnt equal", bool((outs[0][2] == outs[1][2]).all()))

@triton.jit
def routing3(logit_ptr, bias_ptr, local_start_ptr, ids_ptr, w_ptr, rowmap_ptr, cnt_ptr, slab, T,
             LOGIT_STRIDE: tl.constexpr, HILO: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    t64 = t.to(tl.int64)
    e = tl.arange(0, 256)
    x = tl.load(logit_ptr + t64[:, None] * LOGIT_STRIDE + e[None, :], mask=tm[:, None], other=0.0)
    if HILO:
        x += tl.load(logit_ptr + t64[:, None] * LOGIT_STRIDE + 256 + e[None, :], mask=tm[:, None], other=0.0)
    sc = tl.sigmoid(x * 0.1767766952966369)
    b = sc + tl.load(bias_ptr + e)[None, :]
    j = tl.arange(0, 8)
    ids = tl.zeros([BT, 8], dtype=tl.int32)
    ws = tl.zeros([BT, 8], dtype=tl.float32)
    for k in tl.static_range(8):
        m = tl.max(b, 1)
        idx = tl.min(tl.where(b == m[:, None], e[None, :], 256), 1)
        hit = e[None, :] == idx[:, None]
        wk = tl.sum(tl.where(hit, sc, 0.0), 1)
        ids = tl.where(j[None, :] == k, idx[:, None], ids)
        ws = tl.where(j[None, :] == k, wk[:, None], ws)
        b = tl.where(hit, float("-inf"), b)
    ws = ws / (tl.sum(ws, 1)[:, None] + 1e-15)
    o = t64[:, None] * 8 + j[None, :]
    m2 = tm[:, None]
    tl.store(ids_ptr + o, ids, mask=m2)
    tl.store(w_ptr + o, ws, mask=m2)
    ls = tl.load(local_start_ptr)
    le = tl.reshape((ids - ls + 256) % 256, [BT * 8])
    valid = tl.reshape(m2 & (j[None, :] >= 0), [BT * 8])
    le = tl.where(valid, le, 999)
    c = tl.arange(0, 32)
    onehot = (le[:, None] == c[None, :]).to(tl.int32)          # [BT*8, 32]
    incl = tl.cumsum(onehot, 0)
    tot = tl.sum(onehot, 0)                                      # [32]
    base = tl.atomic_add(cnt_ptr + c, tot, mask=tot > 0)         # one atomic per expert per block
    rank = tl.sum(tl.where(onehot > 0, incl - 1 + base[None, :], 0), 1)
    is_local = le < 32
    rm = tl.where(is_local, le * slab + rank, -1)
    tl.store(rowmap_ptr + tl.reshape(o, [BT * 8]), rm, mask=valid)

print("---- skewed logits (one hot expert) ----")
for T in [753, 3012, 16384]:
    logits = torch.randn(T, 512, device=dev)
    logits[:, 3] += 30; logits[:, 7] += 10; logits[:, 12] += 5
    bias = torch.randn(256, device=dev) * 0.05
    ls = torch.tensor([0], dtype=torch.int32, device=dev)
    res = {}
    for name, launch in [
        ("old", lambda ids, tw, rm, cnt: sub._routing_kernel[(T,)](logits, bias, ls, ids, tw, rm, cnt, T, LOGIT_STRIDE=512, HILO=True, num_warps=2)),
        ("agg bt16", lambda ids, tw, rm, cnt: routing3[(triton.cdiv(T, 16),)](logits, bias, ls, ids, tw, rm, cnt, T, T, LOGIT_STRIDE=512, HILO=True, BT=16, num_warps=4)),
        ("agg bt32", lambda ids, tw, rm, cnt: routing3[(triton.cdiv(T, 32),)](logits, bias, ls, ids, tw, rm, cnt, T, T, LOGIT_STRIDE=512, HILO=True, BT=32, num_warps=4)),
        ("agg bt64", lambda ids, tw, rm, cnt: routing3[(triton.cdiv(T, 64),)](logits, bias, ls, ids, tw, rm, cnt, T, T, LOGIT_STRIDE=512, HILO=True, BT=64, num_warps=8)),
    ]:
        ids = torch.empty(T, 8, dtype=torch.int32, device=dev); tw = torch.empty(T, 8, device=dev)
        rm = torch.empty(T, 8, dtype=torch.int32, device=dev); cnt = torch.zeros(32, dtype=torch.int32, device=dev)
        t = timeit(lambda: (cnt.zero_(), launch(ids, tw, rm, cnt)))
        cnt.zero_(); launch(ids, tw, rm, cnt); torch.cuda.synchronize()
        # validate rowmap: every local route has a unique row within [e*slab, e*slab+cnt[e])
        loc = rm[rm >= 0]
        e_of = loc // T; r_of = loc % T
        ok = bool((r_of < cnt[e_of]).all()) and loc.unique().numel() == loc.numel()
        res[name] = (ids.clone(), cnt.clone())
        print(T, name, f"{t:.1f} us", "rowmap ok" if ok else "ROWMAP BAD", "cnt==old" if torch.equal(cnt, res["old"][1]) else "CNT DIFF")
