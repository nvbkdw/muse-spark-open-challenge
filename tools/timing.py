import importlib.util, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "muse_bf16_proj"))
import reference
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
w = reference.generate_weights(reference.WEIGHTS_SEED); st = sub.prepare(w)
flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
for s in sys.argv[2:]:
    layer, m = s.split(":"); m = int(m)
    data = reference.generate_input(m, layer, seed=7)
    for _ in range(3): sub.kernel_fn(data, st)
    flush.zero_(); st["tdbg"].zero_(); torch.cuda.synchronize()
    sub.kernel_fn(data, st); torch.cuda.synchronize()
    t = st["tdbg"].view(-1, 8)[:132].double()
    t0 = t[:, 0].min()
    rel = (t - t0) / 1000.0
    names = ["start", "excdone", "first_full", "last_full", "loop_end", "flush_end"]
    print(layer, m, " ".join(f"{nm}: med {rel[:, i].median():.2f} max {rel[:, i].max():.2f}" for i, nm in enumerate(names)))
    sm = st["tdbg"].view(-1, 8)[:132, 6]
    print("  distinct SMs used:", int(torch.unique(sm).numel()))
