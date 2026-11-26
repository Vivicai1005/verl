"""Correctness of TE's CK BF16 grouped GEMM against the per-expert hipBLASLt path (same inputs, same weights).

GroupedLinear, 32 experts, fuse_wgrad_accumulation=True (wgrad accumulated into an FP32 main_grad, as Megatron does),
FC1 2048->1536 and FC2 768->2048, uneven token splits. Compares the forward output, dgrad and wgrad.
"""

import os

import torch
import transformer_engine.pytorch as te


def run(H, F, ck):
    for k in ("NVTE_USE_GROUPED_GEMM_TRITON", "NVTE_USE_HIPKITTENS_GROUPED_GEMM", "NVTE_USE_CK_GROUPED_GEMM"):
        os.environ.pop(k, None)
    if ck:
        os.environ["NVTE_USE_CK_GROUPED_GEMM"] = "1"
    torch.manual_seed(0)
    E = 32
    m = te.GroupedLinear(E, H, F, bias=False, params_dtype=torch.bfloat16, device="cuda", fuse_wgrad_accumulation=True)
    for p in m.parameters():
        p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device="cuda")
    splits = torch.randint(200, 900, (E,)).tolist()
    x = torch.randn(sum(splits), H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = m(x, splits)
    y.backward(torch.randn_like(y))
    torch.cuda.synchronize()
    return y.float(), x.grad.float(), torch.stack([p.main_grad for p in m.parameters()])


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


for H, F in ((2048, 1536), (768, 2048)):
    ref = run(H, F, ck=False)
    got = run(H, F, ck=True)
    errs = [rel(g, r) for g, r in zip(got, ref, strict=False)]
    print(
        f"{H}->{F}: relative error CK vs hipBLASLt  fprop {errs[0]:.2e}  dgrad {errs[1]:.2e}  wgrad {errs[2]:.2e}  "
        f"{'OK' if max(errs) < 1e-2 else 'MISMATCH'}"
    )
