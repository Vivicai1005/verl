"""Pad routed-expert tokens to 256 for MXFP8 on ROCm.

Megatron pads each expert's tokens to get_align_size_for_quantization(), which is 32 for MXFP8 (the NVIDIA
requirement). On gfx950, Transformer Engine runs MXFP8 GEMMs on HipKittens, both the per-expert dense kernels and the
grouped kernels (NVTE_USE_HIPKITTENS_GROUPED_GEMM=1), and HipKittens needs the token dimension to be a multiple of
256. An expert whose token count misses that falls back to hipBLASLt, which has no prebuilt kernel for the
non-swizzled MXFP8 scales Transformer Engine passes and generates one at run time with rocRoller (minutes per new
kernel); below 128 Transformer Engine's K % 128 check also fails in the expert weight-gradient GEMM. Only the token
padding changes: get_fp8_align_size() keeps returning 32, because other callers (GatedDeltaNet's in_proj width check)
use it for dimensions that are not token counts.

Idempotent; also upgrades an install patched by the earlier 128 version of this script.
"""

import pathlib

import megatron.core.transformer.moe.moe_utils as moe_utils

path = pathlib.Path(moe_utils.__file__)
original = """    if config.fp8:
        return get_fp8_align_size(config.fp8_recipe)
"""
patched_128 = """    if config.fp8:
        # ROCm Transformer Engine's MXFP8 GEMM needs K % 128 == 0; K is the per-expert token count in the
        # expert weight-gradient GEMM.
        recipe = getattr(config.fp8_recipe, "value", config.fp8_recipe)
        if torch.version.hip is not None and recipe == "mxfp8":
            return 128
        return get_fp8_align_size(config.fp8_recipe)
"""
new = """    if config.fp8:
        # ROCm Transformer Engine runs MXFP8 GEMMs on HipKittens, which needs each expert's token count to be a
        # multiple of 256; otherwise the GEMM falls back to hipBLASLt/rocRoller (and below 128 fails TE's K check).
        recipe = getattr(config.fp8_recipe, "value", config.fp8_recipe)
        if torch.version.hip is not None and recipe == "mxfp8":
            return 256
        return get_fp8_align_size(config.fp8_recipe)
"""
src = path.read_text()
if new in src:
    print("already patched", path)
else:
    for old in (patched_128, original):
        if src.count(old) == 1:
            path.write_text(src.replace(old, new))
            print("patched", path)
            break
    else:
        raise AssertionError("megatron get_align_size_for_quantization patch target not found")
