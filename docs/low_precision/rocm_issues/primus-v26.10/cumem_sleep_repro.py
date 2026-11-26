"""Minimal repro of vLLM sleep mode on ROCm: allocate tensors through vLLM's CuMemAllocator pool, sleep, wake.

No model: this isolates vLLM's cumem allocator (csrc/cumem_allocator.cpp, hipMemCreate/Map/Unmap/Release) from verl.
"""

import torch
import vllm
from vllm.device_allocator.cumem import CuMemAllocator

print("vllm", vllm.__version__, "| torch", torch.__version__, "| hip", torch.version.hip)
alloc = CuMemAllocator.get_instance()
with alloc.use_memory_pool(tag="weights"):
    w = torch.randn(64 * 1024 * 1024, device="cuda")  # 256 MiB
with alloc.use_memory_pool(tag="kv_cache"):
    kv = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # 512 MiB
ref = w[:8].clone()
torch.cuda.synchronize()
for level, tags in ((1, ("weights",)), (2, None)):
    try:
        alloc.sleep(offload_tags=tags)
        torch.cuda.synchronize()
        alloc.wake_up()
        ok = level == 2 or torch.equal(w[:8], ref)
        print(f"sleep(offload_tags={tags}) + wake_up: OK, weights preserved={ok}")
    except Exception as e:
        print(f"sleep(offload_tags={tags}): FAILED: {e}")
        break
