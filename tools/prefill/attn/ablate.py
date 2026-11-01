import sys, torch, os
from common import *
import attn_impl as A
orig = A._cuda_source
abl = {
 "base": [],
 "noexp": [("EXP_FN(x)", "(x)")],
 "nomax": [("float mx0 = fmaxf(xa, xb), mx1 = fmaxf(ya, yb);", "float mx0 = 30.f, mx1 = 30.f; (void)xa; (void)xb; (void)ya; (void)yb;")],
 "nopack": [("pa[4 * c + 0] = pack4(x[0], x[1], x[4], x[5]);", "pa[4 * c + 0] = __float_as_uint(x[0]+x[1]+x[4]+x[5]);"),
            ("pa[4 * c + 1] = pack4(x[2], x[3], x[6], x[7]);", "pa[4 * c + 1] = __float_as_uint(x[2]+x[3]+x[6]+x[7]);"),
            ("pa[4 * c + 2] = pack4(x[8], x[9], x[12], x[13]);", "pa[4 * c + 2] = __float_as_uint(x[8]+x[9]+x[12]+x[13]);"),
            ("pa[4 * c + 3] = pack4(x[10], x[11], x[14], x[15]);", "pa[4 * c + 3] = __float_as_uint(x[10]+x[11]+x[14]+x[15]);")],
 "nopv": [("for (int c = 0; c < 4; c++) mma_pv(o,", "for (int c = 0; c < 0; c++) mma_pv(o,"), ("for (int c = 0; c < 4; c++) mma_rs(rs,", "for (int c = 0; c < 0; c++) mma_rs(rs,"), ("o[i] *= a0; o[i + 1] *= a0;", "o[i] += __uint_as_float(pa[i/4]); o[i + 1] *= a0;")],
 "nors": [("for (int c = 0; c < 4; c++) mma_rs(rs,", "for (int c = 0; c < 0; c++) mma_rs(rs,")],
}
name = sys.argv[1]
def patched():
    s = orig()
    for a, b in abl[name]:
        assert a in s, a
        s = s.replace(a, b)
    return s
A._cuda_source = patched
c = make_case(1, 32768, 16384, 0, 1)
f2 = lambda: A.custom_attention(c["q"], c["k_cache"], c["v_cache"], c["page_table"], c["cache_seqlens"], 16384, SCALE, c["k_scale"], c["v_scale"])
t = timeit(f2)
print(f"{name}: {t:.0f} us  ({flops(c)/t/1e6:.0f} TF/s)")
