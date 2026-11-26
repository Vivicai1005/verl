"""Experiment: make vLLM 0.34 cumem CPU backups 0.32-style (torch pinned tensor, no mmap + cudaHostRegister)."""

import sys

p = sys.argv[1]
s = open(p).read()
a = "def create_cpu_backup(size_in_bytes: int) -> torch.Tensor:\n"
b = "def release_cpu_backup(backup: torch.Tensor) -> None:\n"
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, a + '    return torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=PIN_MEMORY)\n')
s = s.replace(b, b + "    return\n")
open(p, "w").write(s)
print("pinned-backup patch applied")
