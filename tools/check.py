"""Correctness sweep with the problem's reference checks (scored shapes, judge tests, nearby shapes)."""
import importlib.util, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "muse_bf16_proj"))
import reference
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "muse_bf16_proj", "submission.py")
spec = importlib.util.spec_from_file_location("sub", path); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
seed = int(os.environ.get("WSEED", reference.WEIGHTS_SEED))
weights = reference.generate_weights(seed)
st = sub.prepare({k: v.clone() for k, v in weights.items()})
cases = [(m, l, "normal") for m in (1, 8, 16, 32, 128, 256) for l in reference.LAYERS]
cases += [(1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"), (130, "unpooler", "normal"),
          (5, "unpooler", "normal"), (8, "router", "normal"), (9, "router", "outliers"), (512, "qkv_gate", "outliers"),
          (1000, "unpooler", "outliers")]
cases += [(m, l, d) for m in (2, 3, 4, 9, 12, 15, 17, 24, 25, 31, 40, 64, 100, 200, 300) for l in reference.LAYERS for d in ("normal", "outliers")]
bad = 0
for i, (m, l, d) in enumerate(cases):
    data = reference.generate_input(m, l, seed=1000 + i, dist=d)
    for rep in range(2):  # run twice: workspace/counters must be clean after each call
        out = sub.kernel_fn(data, st)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, out, weights)
    if not ok or "-v" in sys.argv:
        print(("OK  " if ok else "FAIL"), m, l, d, msg)
    bad += not ok
print("cases", len(cases), "failures", bad)
