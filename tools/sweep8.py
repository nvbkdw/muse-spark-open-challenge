STREAM_SIZES = []
ROTS = (7,)
SWEEP = []
def add(layer, mbs, cfgs, var="x"):
    for c in cfgs:
        for mb in mbs:
            SWEEP.append((layer, var, mb, c))
# (WT, WK, D, CL, XK, NSLOT)
QKV = [(3, 4, 4, 2, 16, 2), (3, 8, 2, 2, 8, 4), (3, 8, 4, 2, 8, 4), (6, 4, 2, 4, 8, 4), (6, 4, 4, 4, 8, 4),
       (2, 8, 2, 1, 8, 4), (2, 8, 4, 1, 16, 2), (3, 4, 6, 2, 8, 4)]
add("qkv_gate", [1, 2], QKV)
add("qkv_gate", [1], QKV, var="m8")
add("o", [4], [(4, 4, 2, 1, 8, 2), (2, 8, 2, 2, 8, 2), (4, 4, 4, 1, 8, 2)])
add("unpooler", [4], [(4, 4, 2, 1, 16, 2), (4, 4, 2, 1, 8, 4), (8, 2, 2, 2, 8, 4)])
add("pooler", [1], [(4, 4, 2, 2, 8, 4), (4, 8, 2, 2, 8, 4)])
