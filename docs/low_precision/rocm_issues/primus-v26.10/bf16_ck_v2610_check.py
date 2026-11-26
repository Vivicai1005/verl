"""Which kernels run the BF16 MoE experts in verl-qat:primus-v26.10 (TE 2.21 with ROCm/TE #783), and how fast?

TE GroupedLinear as on one EP4 rank of Qwen3-30B-A3B: 32 local experts, uneven tokens (about 20k per micro-batch),
one forward + backward. FC1 2048->1536 (gate+up) and FC2 768->2048. Megatron-Bridge sets
fuse_wgrad_accumulation=True for the actor (wgrad accumulated into an FP32 main_grad).
Cases (TE reads these env vars per call):
  script_bf16 : what the ROCm script's bf16 branch exports today (NVTE_USE_GROUPED_GEMM_TRITON=1)
  ck          : NVTE_USE_CK_GROUPED_GEMM=1
  ck_nofuse   : NVTE_USE_CK_GROUPED_GEMM=1 with fuse_wgrad_accumulation=False (in case the ref is built that way)
"""

import os
import re
from collections import defaultdict

import torch
import transformer_engine as te_pkg
import transformer_engine.pytorch as te

KEYS = ("NVTE_USE_GROUPED_GEMM_TRITON", "NVTE_USE_HIPKITTENS_GROUPED_GEMM", "NVTE_USE_CK_GROUPED_GEMM")


def run(case, H, F, fuse, env):
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    torch.manual_seed(0)
    E = 32
    m = te.GroupedLinear(E, H, F, bias=False, params_dtype=torch.bfloat16, device="cuda", fuse_wgrad_accumulation=fuse)
    if fuse:
        for p in m.parameters():
            p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device="cuda")
    splits = torch.randint(300, 1000, (E,)).tolist()
    x = torch.randn(sum(splits), H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    for _ in range(3):  # warm-up (JIT, tile selection)
        m(x, splits).sum().backward()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            m(x, splits).sum().backward()
        torch.cuda.synchronize()
    agg, cnt = defaultdict(float), defaultdict(int)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name] += e.device_time / 5
            cnt[e.name] += 1
    total = sum(agg.values())
    print(f"===== {case} {H}->{F} tokens={sum(splits)} fuse_wgrad={fuse} env={env}: {total / 1000:.3f} ms per fwd+bwd")
    for n, t in sorted(agg.items(), key=lambda x: -x[1])[:5]:
        print(f"  {t / 1000:7.3f} ms  x{cnt[n] // 5:<4d} {re.sub(r'_UserArgs.*', '_...', n)[:120]}")


print("TE", te_pkg.__version__, "| torch", torch.__version__)
for H, F in ((2048, 1536), (768, 2048)):
    run("script_bf16", H, F, True, {"NVTE_USE_GROUPED_GEMM_TRITON": "1"})
    run("ck", H, F, True, {"NVTE_USE_CK_GROUPED_GEMM": "1"})
    run("ck_nofuse", H, F, False, {"NVTE_USE_CK_GROUPED_GEMM": "1"})
