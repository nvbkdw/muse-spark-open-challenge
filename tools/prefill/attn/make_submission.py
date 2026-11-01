"""Build tools/prefill/attn/submission_attn.py = muse_prefill_layer/submission.py with the FA3 call replaced by
attn_impl.attention (whole attn_impl source pasted in: single file, no extra imports)."""
import os
import re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sub = open(os.path.join(ROOT, "muse_prefill_layer", "submission.py")).read()
impl = open(os.path.join(HERE, "attn_impl.py")).read()
body = impl[impl.find("FP8 = torch.float8_e4m3fn"):].replace("FP8 = torch.float8_e4m3fn\n", "")
body = re.sub(r"CFG = dict\(.*?\)\)\n", "CFG = dict(ROWSUM=2, MAXTREE=1, POLY=0, EXPFN='ex2', F16SM=0)\n", body, flags=re.S)
assert "CFG = dict(ROWSUM=2" in body
header = ("\n# ----------------------------------------------------------------------------------------\n"
          "# Attention: custom FP8 GQA-packed wgmma kernel (NVRTC, driver-API launch) for full layers,\n"
          "# FA3 for sliding-window layers. Source: tools/prefill/attn/attn_impl.py\n"
          "# ----------------------------------------------------------------------------------------\n"
          "import ctypes  # noqa: E402\nimport math  # noqa: E402\n")
sub = sub.replace("HIDDEN = 8192\n", header + body + "\n\nHIDDEN = 8192\n", 1)
a = sub.find('    window = cfg["window"]\n    descale = {}')
b = sub.find("    o_in = torch.empty((T, 1024)")
assert a > 0 and b > a
sub = sub[:a] + '''    window = cfg["window"]
    out = attention(q8, kc, vc, data["page_table"], data["cache_seqlens"], data["cu_seqlens_q"], length,
                    cfg["sm_scale"] * cfg["qk_scale_factor"], window, data["k_scale"], data["v_scale"],
                    kv_scale_is_one=cfg["kv_scale"] == 1.0, scheduler_metadata=data["scheduler_metadata"])
''' + sub[b:]
sub = sub.replace("    _buffers(st, 16384, dev)\n", "    _buffers(st, 16384, dev)\n    _compile()  # NVRTC-compile the attention kernel (~1 s)\n", 1)
open(os.path.join(HERE, "submission_attn.py"), "w").write(sub)
print("wrote submission_attn.py")
