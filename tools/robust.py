"""Robustness: other weight seeds and adversarial weight distributions (exactness / fallback paths)."""
import importlib.util, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "muse_bf16_proj"))
import reference
spec = importlib.util.spec_from_file_location("sub", os.path.join(HERE, "..", "muse_bf16_proj", "submission.py"))
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)

def variants():
    for seed in (1, 2, 12345):
        yield f"seed{seed}", reference.generate_weights(seed)
    w = reference.generate_weights(7)
    w2 = dict(w); w2["qkv_gate"] = (w["qkv_gate"].float() * 1000).to(torch.bfloat16)      # shifted exponent window
    w2["o"] = w["o"].clone(); w2["o"][::7, ::5] = 0                                        # many exact zeros (exceptions)
    yield "scaled+zeros", w2
    w3 = dict(w); w3["pooler"] = (torch.randn_like(w["pooler"].float()) * torch.exp(torch.randn_like(w["pooler"].float()) * 3)).to(torch.bfloat16)  # very wide exponent range -> fallback
    yield "wide-range", w3

bad = 0
for name, w in variants():
    st = sub.prepare({k: v.clone() for k, v in w.items()})
    kinds = {k: v[0] for k, v in st["packs"].items()}
    for i, (m, layer) in enumerate([(m, l) for m in (1, 8, 16, 32, 128, 256, 5, 33) for l in reference.LAYERS]):
        data = reference.generate_input(m, layer, seed=99 + i, dist="outliers" if i % 3 == 0 else "normal")
        out = sub.kernel_fn(data, st); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, out, w)
        if not ok:
            bad += 1; print("FAIL", name, m, layer, msg)
    print(name, "packs:", sorted(set(kinds.values())), "fallback packs:", [k[:2] for k, v in kinds.items() if v == "raw"])
print("failures", bad)
