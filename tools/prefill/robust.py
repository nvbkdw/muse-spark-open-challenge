"""Robustness: kv_scale=1/64, odd batch/length shapes, page 64, spread lengths."""
import sys, os, importlib.util, torch
sys.path.insert(0, "muse_prefill_layer")
import reference as R
spec = importlib.util.spec_from_file_location("sub", "muse_prefill_layer/submission.py"); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
w = R.generate_weights(R.WEIGHTS_SEED)
st = sub.prepare({k: v.clone() for k, v in w.items()})
cases = [
    dict(batch=1, ctx=3000, layer="sliding", q_len=753, page=1, kv_scale=1 / 64),
    dict(batch=1, ctx=40000, layer="full", q_len=1000, page=1, kv_scale=1 / 64, dist="soft"),
    dict(batch=2, ctx=20000, layer="full", q_len=1500, page=1, spread="agentic", dist="soft"),
    dict(batch=3, ctx=7000, layer="sliding", q_len=600, page=64, spread="mixed"),
    dict(batch=1, ctx=10000, layer="full", q_len=5000, page=1, dist="soft"),
]
for i, c in enumerate(cases):
    c = c | {"seed": 99 + i}
    ls = R.case_fields(c, w)["local_start"]
    d = R.generate_input(**c, local_start=ls)
    out = sub.kernel_fn(d, st)
    ok, msg = R.check_implementation(d, R.collect(d, out), w)
    print(("PASS" if ok else "FAIL"), {k: v for k, v in c.items() if k != "seed"}, msg[:200], flush=True)
