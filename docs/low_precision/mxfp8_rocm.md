# MXFP8 RL on AMD ROCm

Last updated: 11/26/2026

AMD Instinct MI350 / MI355 (gfx950) decode the OCP MXFP8 format natively: E4M3 elements with one E8M0 scale per
32-element block. This page records running the [MXFP8 RL](mxfp8.md) recipe, Transformer Engine's MXFP8 training
recipe with an MXFP8 rollout, on gfx950, model by model. The rollout quantization interface is described in
[Low-Precision Rollout and Quantization-Aware Training](qat.md).

How the ROCm settings below were chosen, the comparisons behind them, and where the time goes are in
[MXFP8 RL on AMD ROCm: performance notes](mxfp8_rocm_optimization.md).

## Requirements

- AMD Instinct MI350 / MI355 (gfx950)
- The `docker/rocm/Dockerfile.rocm.primus-v26.10` image: `rocm/primus:v26.10` (ROCm "TheRock" SDK 10.3, PyTorch 2.16,
  Transformer Engine 2.21.0.dev0, ROCm fork at `b5c00f432`), Megatron-Core core_v0.19.2, Megatron-Bridge v0.6.2,
  vLLM v0.34.0
- The three vLLM 0.34 patches in `docker/rocm/patches/`, applied in the container in this order (the Dockerfile
  leaves the stack stock and lists what each one fixes):
  `vllm-0.34-rocm-mxfp8-block32-scales-switch.patch`, `vllm-0.34-rocm-cumem-pinned-backup.patch`,
  `vllm-0.34-rocm-mxfp8-linear-opaque-op.patch`
- `NVTE_ROCM_ENABLE_MXFP8=1`, which enables Transformer Engine's MXFP8 path on gfx95x

## Configuration

The same as on Blackwell (see [MXFP8 RL](mxfp8.md#configuration)), plus the ROCm Megatron attention backend:

```yaml
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "e4m3"
  fp8_recipe: "mxfp8"
  attention_backend: fused   # the image exports NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=1

actor_rollout_ref.rollout.quantization:
  linear: MXFP8
  moe: MXFP8
```

Notes:

- On gfx950 Transformer Engine runs an MXFP8 GEMM on HipKittens only when its token and feature dimensions are
  multiples of 256 (K a multiple of 128); otherwise rocRoller generates a kernel on the CPU at run time, minutes per
  kernel. verl pads each expert's tokens to 256 for MXFP8 on ROCm (`moe_router_padding_for_fp8=True`); the example
  script sets the remaining options (below).
- The reference model inherits the actor's MXFP8 settings, as on Blackwell.
- `VLLM_ROCM_MXFP8_BLOCK32_SCALES=0` (exported by the example script) keeps vLLM's per-row MXFP8 weight scales, so the
  weight refit works after vLLM starts on dummy weights.
- `lm_head`, the token embedding and MoE routers stay in BF16 on the rollout, as on Blackwell.

## Rollout kernels

The MXFP8 dense linears run on vLLM's gfx950 kernel (`RocmDotScaledMxfp8LinearKernel`, Triton `tl.dot_scaled`). With
`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1` the MoE runs on AITER, whose MXFP8 MoE uses FlyDSL kernels.

## Example: Qwen3-30B-A3B on DAPO-Math-17k

`examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh` is the ROCm version of the
[Blackwell example](mxfp8.md#example-qwen3-30b-a3b-on-dapo-math-17k) for one 4×MI355X node: GRPO on Qwen3-30B-A3B
(MoE, 128 experts, top-8) with DAPO-Math-17k, MXFP8 end to end, experts included; `PRECISION=bf16` runs the same job
in BF16. The task settings are the upstream ones: 64 prompts × 8 responses per step, prompts and responses up to 1024
and 8192 tokens, dynamic batching at 20,480 tokens per GPU, learning rate 1e-6, KL loss 0.001 (`low_var_kl`), no
validation set. The script reads DAPO-Math-17k in verl's format from `TRAIN_FILE`.

```bash
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1   # rollout MoE on AITER, as in the results below
COMMON=(SEED=42 TOTAL_TRAINING_STEPS=10 MODEL_PATH=Qwen/Qwen3-30B-A3B TRAIN_FILE=$HOME/data/dapo-math-17k/train.parquet)
env "${COMMON[@]}" PRECISION=mxfp8 bash examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh
env "${COMMON[@]}" PRECISION=bf16 bash examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh
```

`SEED` fixes the data order so steps pair across runs. The script's defaults fit one 4×MI355X node: training
TP1 × EP4 × ETP1 with selective recompute of `core_attn` and `moe_act` and the whole optimizer on the CPU; the rollout
on vLLM with TP2 and `gpu_memory_utilization=0.4`. By precision it sets:

| `PRECISION` | Settings |
|---|---|
| `mxfp8` | `NVTE_ROCM_ENABLE_MXFP8=1`; experts on HipKittens' grouped MXFP8 GEMM (`NVTE_USE_HIPKITTENS_GROUPED_GEMM=1`); `moe_router_padding_for_fp8=True`; `pad_to_length=True` with `PAD_TO_LENGTH_BUCKET=512` for the dense GEMMs; the reference model in MXFP8 too |
| `bf16` | experts on Transformer Engine's CK grouped GEMM (`NVTE_USE_CK_GROUPED_GEMM=1`) |

### Results

**Configuration**

- One 4×MI355X machine, the same four GPUs for every run, back to back; `SEED=42`, so steps pair across runs; 10 steps
- The script's defaults (above); the rollout MoE on AITER (`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`)
- The `docker/rocm/Dockerfile.rocm.primus-v26.10` image with its three vLLM 0.34 patches
- DAPO-Math-17k deduplicated to its 17,398 unique prompts
- W&B: [wei-cai/qwen3-30b-a3b_dapo_mi355](https://wandb.ai/wei-cai/qwen3-30b-a3b_dapo_mi355), runs
  `bf16ck_v2610_aiter2` and `mxfp8_v2610_aiter_opq3`

"X/Y" is training precision X, rollout precision Y. Step 1 is left out of the means: it includes one-time loading and
compilation (473 s for BF16, 485 s for MXFP8).

| mean of steps 2–10 | BF16/BF16 | MXFP8/MXFP8 |
|---|---|---|
| step time (s) | 354.7 | 353.3 |
| generation (s) | 156.7 | 143.4 |
| `old_log_prob` (s) | 39.5 | 40.3 |
| ref (s) | 39.3 | 39.9 |
| `update_actor` (s) | 116.4 | 123.9 |
| `update_weights` (s) | 2.4 | 5.3 |
| throughput (tokens/s/GPU) | 2309 | 2321 |
| `rollout_corr/kl` | 0.0018 | 0.0049 |
| score | −0.079 | −0.075 |
| peak memory (GB) | 188 | 198 |

Results and observations:

- MXFP8 matches BF16 end to end on 4×MI355X: the step takes 0.4% less and throughput is 0.5% higher. Generation is
  8.5% shorter (against −24% on B200), while `update_actor` takes 6.5% longer and `update_weights` 3 s longer.
- The rollout–trainer mismatch is 2.7× BF16's (0.0049 against 0.0018), close to the ratio on B200 (0.0045 against
  0.0018).
- The machine is shared, and host-memory reclaim by other jobs slowed the training phases of a few steps. Over the
  steps where neither run saw reclaim (3, 5 and 6) the step takes 352.4 s for BF16 and 349.5 s for MXFP8.
- Ten steps are enough for speed and mismatch, not for the score.

## Example: Qwen3.5-35B-A3B

Qwen3.5-35B-A3B (MoE, 40 layers, GatedDeltaNet and full attention 3:1, 256 routed experts) cannot use Transformer
Engine's MXFP8 training recipe on gfx950 yet. On gfx950 every MXFP8 GEMM needs its reduction dimension K to be a
multiple of 128 (32 on NVIDIA GPUs and gfx1250), and the fused GatedDeltaNet `in_proj` of Qwen3.5 is 12,352 wide
(6,176 per rank at TP2), which is not a multiple of 128 at any TP size. See
[the performance notes](mxfp8_rocm_optimization.md#qwen35-35b-a3b-mxfp8-gemm-k--128-on-gfx950) for the details.
