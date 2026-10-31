# okc: problem=muse_bf16_proj
"""Probe: baseline-equivalent cuBLAS path + environment printout (check run only)."""
import torch, sys, os, shutil
import torch.nn.functional as F

def prepare(weights):
    try:
        import triton; tv = triton.__version__
    except Exception as e:
        tv = repr(e)
    print("ENV torch", torch.__version__, "cuda", torch.version.cuda, "triton", tv, "py", sys.version.split()[0])
    print("ENV gpu", torch.cuda.get_device_name(), torch.cuda.get_device_properties(0))
    print("ENV nvcc", shutil.which("nvcc"), os.environ.get("CUDA_HOME"), "ninja", shutil.which("ninja"))
    st = dict(weights)
    r = weights["router"]
    hi = r.to(torch.bfloat16); lo = (r - hi.float()).to(torch.bfloat16)
    st["router_hi_t"] = hi.t().contiguous(); st["router_lo_t"] = lo.t().contiguous()
    return st

def kernel_fn(data, state):
    x, layer = data["x"], data["config"]["layer"]
    if layer != "router":
        return F.linear(x, state[layer][: data["config"]["n"]])
    if x.shape[0] <= 8:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
