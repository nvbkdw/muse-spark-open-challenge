import importlib.util, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "muse_bf16_proj"))
import reference
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
w = reference.generate_weights(reference.WEIGHTS_SEED); st = sub.prepare(w)
flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
names = {0: "start", 1: "contrib_rel", 2: "xready", 3: "poll_ok", 4: "loop_end", 5: "flush_end", 7: "prefetch_done"}
for s in sys.argv[2:]:
    layer, m = s.split(":"); m = int(m)
    data = reference.generate_input(m, layer, seed=7)
    for _ in range(3): sub.kernel_fn(data, st)
    flush.zero_(); st["tdbg"].zero_(); torch.cuda.synchronize()
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    t = st["tdbg"].view(-1, 8)[:132].double()
    t0 = t[:, 0].min()
    out = []
    for i, nm in names.items():
        col = t[:, i]; v = col[col > 0]
        if len(v): out.append(f"{nm} {((v - t0) / 1000).median():.2f}/{((v - t0) / 1000).max():.2f}(n{len(v)})")
    print(layer, m, " ".join(out))
