"""Time kernel_fn for shapes via CUDA graph replays, warm and cold L2 (diagnostic). usage: micro.py sub.py layer:m ..."""
import importlib.util, os, sys, statistics, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "muse_bf16_proj"))
import reference
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = sub.prepare({k: v.clone() for k, v in w.items()})
flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
junk = torch.zeros(1, device="cuda")
eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg): junk.add_(1)
def t(g, cold):
    if cold: flush.sum()
    e0.record(); g.replay(); e1.record(); e1.synchronize(); return e0.elapsed_time(e1) * 1e3
fixed = statistics.median(t(eg, True) for _ in range(30))
for s in sys.argv[2:]:
    layer, m = s.split(":"); m = int(m)
    data = reference.generate_input(m, layer, seed=7)
    for _ in range(3): sub.kernel_fn(data, st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = sub.kernel_fn(data, st)
    g.replay(); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, out, w)
    warm = statistics.median(t(g, False) for _ in range(50)) - fixed
    cold = statistics.median(t(g, True) for _ in range(30)) - fixed
    print(f"{layer:9s} m={m:4d} warm {warm:6.2f} cold {cold:6.2f} us {'OK' if ok else 'FAIL '+msg}", flush=True)
