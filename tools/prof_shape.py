"""Run kernel_fn for chosen shapes (cold L2 each call) for nsys/ncu capture. usage: prof_shape.py sub.py layer:m [...]"""
import importlib.util, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "muse_bf16_proj"))
import reference
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
w = reference.generate_weights(reference.WEIGHTS_SEED)
st = sub.prepare({k: v.clone() for k, v in w.items()})
flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
iters = int(os.environ.get("ITERS", 5))
for spec_ in sys.argv[2:]:
    layer, m = spec_.split(":"); m = int(m)
    data = reference.generate_input(m, layer, seed=7)
    for _ in range(2): sub.kernel_fn(data, st)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"{layer}:{m}")
    for _ in range(iters):
        flush.zero_()
        sub.kernel_fn(data, st)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
