import importlib.util, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "muse_bf16_proj"))
import reference
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
w = reference.generate_weights(reference.WEIGHTS_SEED); st = sub.prepare(w)
flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
names = {0: "start", 2: "xready", 4: "loop_end", 1: "exc_added", 3: "wk_staged", 6: "pushed", 7: "csync", 5: "end"}
for s in sys.argv[2:]:
    layer, m = s.split(":"); m = int(m)
    data = reference.generate_input(m, layer, seed=7)
    for _ in range(3): sub.kernel_fn(data, st)
    flush.zero_(); st["tdbg"].zero_(); torch.cuda.synchronize()
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    t = st["tdbg"].view(-1, 8)[:132].double()
    t = t[t[:, 0] > 0]
    t0 = t[:, 0].min()
    out = []
    for i, nm in names.items():
        col = t[:, i]; v = col[col > 0]
        if len(v): out.append(f"{nm} {((v - t0) / 1000).median():.2f}/{((v - t0) / 1000).max():.2f}(n{len(v)})")
    print(layer, m, " ".join(out))

    h = st["tdbg"][132 * 8: 132 * 8 + 132 * 4].view(132, 4).double()
    h = h[h[:, 0] > 0]
    if len(h):
        r = (h - t0) / 1000
        print("  helper: start %.2f/%.2f  off_loaded %.2f/%.2f  x_loaded %.2f/%.2f  done %.2f/%.2f" % tuple(v for i in range(4) for v in (r[:, i].median(), r[:, i].max())))
