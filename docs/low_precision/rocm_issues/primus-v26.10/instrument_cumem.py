"""Instrument vLLM 0.34 cumem.py _put_to_sleep: log tag/size/offload/chunks of every unmap failure and keep going."""

import re
import sys

path = sys.argv[1]
src = open(path).read()
old = """        try:
            unmap_and_release(data.handle)
        finally:
            data.is_asleep = True
"""
new = """        try:
            unmap_and_release(data.handle)
            _SLEEPDBG_STATS[(data.tag, offload, "ok")] = _SLEEPDBG_STATS.get((data.tag, offload, "ok"), 0) + 1
        except Exception as e:
            _SLEEPDBG_STATS[(data.tag, offload, "FAIL")] = _SLEEPDBG_STATS.get((data.tag, offload, "FAIL"), 0) + 1
            logger.error("SLEEPDBG FAIL tag=%s ptr=%#x size=%d offload=%s handle_len=%d tail=%s err=%s",
                         data.tag, ptr, data.handle[1], offload, len(data.handle), str(data.handle[3:])[:200], e)
        finally:
            data.is_asleep = True
        logger.warning("SLEEPDBG stats %s", _SLEEPDBG_STATS) if sum(_SLEEPDBG_STATS.values()) % 200 == 0 else None
"""
assert src.count(old) == 1, "anchor not found"
src = src.replace(old, new)
src = re.sub(r"(\nlogger = init_logger\(__name__\)\n)", r"\1_SLEEPDBG_STATS = {}\n", src, count=1)
assert "_SLEEPDBG_STATS = {}" in src, "logger anchor not found"
# print final stats once per sleep() call
src = src.replace(
    """        logger.info(
            "CuMemAllocator: sleep freed""",
    """        logger.warning("SLEEPDBG final stats %s", _SLEEPDBG_STATS)
        logger.info(
            "CuMemAllocator: sleep freed""",
    1,
)
open(path, "w").write(src)
print("instrumented", path)
