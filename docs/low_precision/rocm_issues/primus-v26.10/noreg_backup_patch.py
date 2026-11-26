"""Experiment: keep vLLM 0.34's mmap CPU backup but skip cudaHostRegister (pageable backup)."""

import sys

p = sys.argv[1]
s = open(p).read()
a = "    if PIN_MEMORY:\n        error = libcudart.cudaHostRegister("
assert s.count(a) == 1
s = s.replace(a, "    if False:\n        error = libcudart.cudaHostRegister(")
open(p, "w").write(s)
print("no-register patch applied")
