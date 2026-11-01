"""Quick feasibility benchmark: humming NVFP4 (W4A16 BF16) MoE GEMM at muse_prefill_layer shapes."""
import sys, time, math, json
import torch
sys.path.insert(0, "muse_prefill_layer")
from nvfp4 import random_weights
from humming.layer import HummingLayer
from humming.config import GemmType

torch.manual_seed(0)
dev = "cuda"
E = 32
def transform(self):
    from humming.transform import prepare_layer_config, transform_humming_tensors
    self.weight_schema, tensors = self.weight_schema.convert_humming(
        tensors=self.state_dict(), shape_n_stacks=[self.shape_n], shape_k_stacks=[self.shape_k],
        param_dtype=self.torch_dtype, num_experts=self.num_experts)
    self.input_schema, _ = self.input_schema.convert_humming(
        tensors=self.state_dict(), shape_n_stacks=[self.shape_n], shape_k_stacks=[self.shape_k],
        param_dtype=self.torch_dtype)
    for name, _ in list(self.named_parameters()):
        delattr(self, name)
    for name, tensor in tensors.items():
        setattr(self, name, torch.nn.Parameter(tensor, requires_grad=False))
    self.humming_config = prepare_layer_config(shape_n=self.shape_n, shape_k=self.shape_k,
        weight_schema=self.weight_schema, input_schema=self.input_schema, num_experts=self.num_experts,
        pad_n_to_multiple=1, pad_k_to_multiple=1, torch_dtype=self.torch_dtype, has_bias=False)
    self._humming_metas = {}
    tensors = {name: t for name in ["weight", "weight_scale", "zero_point", "bias", "weight_scale_2"]
               if (t := getattr(self, name, None)) is not None}
    tensors = transform_humming_tensors(self.humming_config, tensors)
    for name, t in tensors.items():
        setattr(self, name, torch.nn.Parameter(t, requires_grad=False))

def make_layer(n, k):
    gen = torch.Generator(device=dev).manual_seed(1)
    w, sf, gs = random_weights(n, k, generator=gen, device=dev, leading=(E,))
    gs = gs * torch.exp(torch.randn(E, device=dev))
    layer = HummingLayer(shape_n=n, shape_k=k, weight_config={"quant_method": "modelopt", "quant_algo": "nvfp4"},
                         num_experts=E, torch_dtype=torch.bfloat16).cuda()
    layer.load_from_tensors({"weight": w, "weight_scale": sf, "weight_scale_2": gs})
    transform(layer)
    return layer, (w, sf, gs)

def timeit(fn, reps=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    ts = []
    for _ in range(reps):
        flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    ts.sort(); return ts[len(ts) // 2]

mode = sys.argv[1] if len(sys.argv) > 1 else "grouped_contiguous"
for (n, k) in [(8192, 4096), (4096, 4096)]:
    layer, _ = make_layer(n, k)
    for routes in [1500, 5600, 30500]:
        # skewed expert counts
        p = torch.distributions.Dirichlet(torch.full((E,), 2.0)).sample().to(dev)
        ids = torch.multinomial(p, routes, replacement=True)
        counts = torch.bincount(ids, minlength=E)
        if mode == "grouped_contiguous":
            layout = torch.nn.functional.pad(counts.cumsum(0), (1, 0)).long()
            x = torch.randn(routes, k, device=dev, dtype=torch.bfloat16)
            out = torch.empty(routes, n, device=dev, dtype=torch.bfloat16)
            cc = {"gemm_type": "grouped_contiguous"}
            fn = lambda: layer(x, outputs=out, expert_layout=layout, compute_config=cc)
        else:
            from humming.testing.data import generate_moe_tensors
            topk = ids.view(-1, 1).int()
            x = torch.randn(routes, k, device=dev, dtype=torch.bfloat16)
            out = torch.empty(routes, n, device=dev, dtype=torch.bfloat16)
            cc = {"gemm_type": "indexed"}
            bm = int(sys.argv[2]) if len(sys.argv) > 2 else 64
            _, _, sorted_ids, expert_ids, ntp = generate_moe_tensors(topk, E, "indexed", block_size_config=bm)
            fn = lambda: layer(x, outputs=out, sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=ntp, top_k=1, compute_config=cc)
        try:
            t = timeit(fn)
        except Exception as ex:
            print("ERR", n, k, routes, repr(ex)[:300]); continue
        fl = 2 * routes * n * k
        by = E * n * k * (0.5 + 1 / 16)
        print(f"{mode} N={n} K={k} routes={routes}: {t:.1f} us  {fl / t / 1e6:.0f} TFLOP/s  weightBW {by / t / 1e3:.0f} GB/s", flush=True)
