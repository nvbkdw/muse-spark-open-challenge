import sys, torch
sys.path.insert(0, "muse_prefill_layer")
import reference as R
w = R.generate_weights(R.WEIGHTS_SEED)
for (b, ctx, layer, L) in [(1, 80000, "sliding", 753), (4, 80000, "sliding", 753), (1, 32768, "sliding", 16384)]:
    case = dict(batch=b, ctx=ctx, layer=layer, q_len=L, page=1, dist="soft", seed=7)
    ls = R.case_fields(case, w)["local_start"]
    d = R.generate_input(**case, local_start=ls)
    fr = R.ref_front(d, w)
    ids = fr["biased"].topk(8, dim=-1).indices
    loc = (ids - ls) % 256
    loc = loc[loc < 32]
    cnt = torch.bincount(loc, minlength=32).sort(descending=True).values
    print(f"T={b*L} local_start={ls} routes={int(cnt.sum())} distinct={int((cnt>0).sum())} counts={cnt.tolist()}")
