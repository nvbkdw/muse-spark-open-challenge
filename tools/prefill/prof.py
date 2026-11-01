"""Per-kernel GPU time for one case (torch.profiler over eager calls after warmup)."""
import sys, os, importlib.util, torch
sys.path.insert(0, "muse_prefill_layer")
import reference as R
from torch.profiler import profile, ProfilerActivity
sub_path = os.environ.get("SUB", "muse_prefill_layer/submission.py")
spec = importlib.util.spec_from_file_location("sub", sub_path); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
w = R.generate_weights(R.WEIGHTS_SEED)
st = sub.prepare({k: v.clone() for k, v in w.items()})
b, ctx, layer, L = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
case = dict(batch=b, ctx=ctx, layer=layer, q_len=L, page=1, dist="soft", seed=11)
ls = R.case_fields(case, w)["local_start"]
d = R.generate_input(**case, local_start=ls)
for _ in range(3): sub.kernel_fn(d, st)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(5): sub.kernel_fn(d, st)
    torch.cuda.synchronize()
evs = {}
for e in prof.events():
    if e.device_type.name == "CUDA":
        evs.setdefault(e.name[:70], []).append(e.device_time)
tot = sum(sum(v) for v in evs.values()) / 5
for k, v in sorted(evs.items(), key=lambda kv: -sum(kv[1]))[:18]:
    print(f"{sum(v)/5:9.1f} us  x{len(v)//5}  {k}")
print(f"{tot:9.1f} us total")
