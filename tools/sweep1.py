STREAM_SIZES = [6.3, 12.6, 28.3, 50.3]
_B = [(8, 2, 4, 2), (8, 2, 6, 2), (8, 2, 8, 2), (8, 2, 8, 4), (16, 1, 4, 2), (16, 1, 6, 2), (16, 2, 4, 2), (8, 4, 4, 2), (8, 2, 4, 1)]
SWEEP = []
for layer in ("qkv_gate", "pooler", "unpooler"):
    for c in _B:
        SWEEP.append((layer, "x", 1, c))
for c in [(8, 2, 4, 2), (8, 2, 6, 2), (16, 1, 4, 2), (16, 2, 4, 2), (8, 4, 4, 2)]:
    SWEEP.append(("qkv_gate", "x", 4, c)); SWEEP.append(("unpooler", "x", 4, c))
for c in [(4, 16, 4, 2), (4, 16, 8, 2), (4, 16, 8, 4), (4, 8, 4, 2), (8, 2, 4, 2)]:
    SWEEP.append(("o", "x", 1, c)); SWEEP.append(("o", "x", 4, c))
for c in [(8, 2, 4, 2), (8, 2, 8, 2), (4, 2, 4, 2), (4, 4, 8, 2), (2, 4, 8, 2)]:
    SWEEP.append(("router", "x", 1, c)); SWEEP.append(("router", "x", 4, c))
