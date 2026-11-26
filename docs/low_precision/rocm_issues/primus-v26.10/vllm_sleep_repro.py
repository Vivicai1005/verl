"""Standalone vLLM sleep/wake repro with verl's rollout engine args (Qwen3-30B-A3B, dummy weights, TP2).

usage: python3 vllm_sleep_repro.py {mxfp8|bf16} [sleep_level] [eager]
"""

import sys

from vllm import LLM, SamplingParams

MXFP8_QCFG = {
    "quant_method": "compressed-tensors",
    "format": "mxfp8-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "dynamic": False,
                "scale_dtype": "torch.uint8",
                "zp_dtype": None,
                "observer": "memoryless_minmax",
                "observer_kwargs": {},
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "dynamic": True,
                "scale_dtype": "torch.uint8",
                "zp_dtype": None,
                "observer_kwargs": {},
            },
            "format": "mxfp8-quantized",
        }
    },
    "ignore": [
        "re:.*lm_head$",
        "re:.*embed_tokens$",
        r"re:.*visual\..*",
        r"re:^mtp\..*",
        r"re:.*mlp\.gate$",
        "re:.*shared_expert_gate$",
    ],
    "kv_cache_scheme": None,
    "sparsity_config": {},
    "global_compression_ratio": None,
}


def main():
    mode = sys.argv[1]
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    eager = len(sys.argv) > 3 and sys.argv[3] == "eager"
    llm = LLM(
        model="/results/models/Qwen3-30B-A3B",
        dtype="bfloat16",
        load_format="dummy",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        max_model_len=40960,
        max_num_seqs=256,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        enable_sleep_mode=True,
        gpu_memory_utilization=0.4,
        seed=42,
        hf_overrides={"quantization_config": MXFP8_QCFG} if mode == "mxfp8" else None,
        compilation_config={"cudagraph_mode": "NONE" if eager else "FULL_AND_PIECEWISE"},
    )
    sp = SamplingParams(max_tokens=8, temperature=0)
    llm.generate(["hello"], sp)
    for i in range(2):
        llm.sleep(level=level)
        print(f"REPRO sleep {i} (level={level}) OK", flush=True)
        llm.wake_up()
        print(f"REPRO wake {i} OK", flush=True)
        llm.generate(["hello"], sp)
    print(f"REPRO RESULT mode={mode} level={level} eager={eager}: PASS", flush=True)


if __name__ == "__main__":
    main()
