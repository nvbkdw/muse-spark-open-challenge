STREAM_SIZES = []
ROTS = (0, 7)
SWEEP = []
def add(layer, mbs, cfgs, var="x"):
    for c in cfgs:
        for mb in mbs:
            SWEEP.append((layer, var, mb, c))
# (WT, WK, D, CL, XK, NSLOT)  -- current best configs, rotation A/B on one node
add("qkv_gate", [1], [(3, 4, 4, 2, 16, 2), (3, 4, 4, 2, 8, 4)])
add("qkv_gate", [1], [(3, 4, 4, 2, 8, 4), (6, 2, 4, 4, 8, 4)], var="m8")
add("pooler", [1, 2, 4], [(4, 4, 2, 2, 8, 4)])
add("pooler", [1], [(4, 4, 2, 2, 8, 4)], var="m8")
add("unpooler", [2, 4], [(4, 4, 2, 1, 16, 2)])
add("unpooler", [1], [(1, 8, 4, 2, 16, 2)])
add("unpooler", [1], [(1, 8, 4, 2, 8, 4)], var="m8")
add("o", [1, 2, 4], [(4, 4, 2, 1, 8, 2)])
add("o", [1], [(4, 4, 2, 1, 8, 2)], var="m8")
add("router", [1, 2, 4], [(1, 8, 4, 4, 16, 4), (1, 8, 4, 4, 32, 2)])
add("router", [1], [(1, 8, 4, 4, 16, 4)], var="m8")
