# MXFP8 RL on AMD ROCm

Last updated: 11/18/2026

AMD Instinct MI350 / MI355 (gfx950) decode the OCP MXFP8 format natively: E4M3 elements with one E8M0 scale per
32-element block. This page records running the [MXFP8 RL](mxfp8.md) recipe, Transformer Engine's MXFP8 training
recipe with an MXFP8 rollout, on gfx950, model by model.

For the MXFP8 rollout with QAT instead of MXFP8 training, see
[Quantization-Aware Training on AMD ROCm](qat_rocm.md). The rollout quantization interface is described in
[Low-Precision Rollout and Quantization-Aware Training](qat.md).

## Requirements

- AMD Instinct MI350 / MI355 (gfx950)
- The `docker/rocm/Dockerfile.rocm.primus-qat` image: `rocm/primus:v26.9` (Transformer Engine 2.20.0.dev0, ROCm fork at
  `023639b5`), Megatron-Core core_v0.19.2, Megatron-Bridge v0.6.2, vLLM v0.32.0
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

## Example: Qwen3-30B-A3B on DAPO-Math-17k

`examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh` is the ROCm version of the
[Blackwell example](mxfp8.md#example-qwen3-30b-a3b-on-dapo-math-17k) for one 8×MI350X node: GRPO on Qwen3-30B-A3B
(MoE, 128 experts, top-8) with DAPO-Math-17k, MXFP8 end to end, experts included; `PRECISION=bf16` runs the same job
in BF16. The task settings are the upstream ones: 64 prompts × 8 responses per step, prompts and responses up to 1024
and 8192 tokens, dynamic batching at 20,480 tokens per GPU, learning rate 1e-6, KL loss 0.001 (`low_var_kl`), no
validation set. The script reads DAPO-Math-17k in verl's format from `TRAIN_FILE`.

```bash
COMMON=(SEED=42 OPTIMIZER_CPU_OFFLOAD=False ACTOR_TP=1 RECOMPUTE=selective TOTAL_TRAINING_STEPS=10
        MODEL_PATH=Qwen/Qwen3-30B-A3B TRAIN_FILE=$HOME/data/dapo-math-17k/train.parquet)
env "${COMMON[@]}" PRECISION=mxfp8 bash examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh
env "${COMMON[@]}" PRECISION=bf16 bash examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh
```

`SEED` fixes the data order so steps pair across runs. The other three variables select the MI350X layout described
below; without them the script keeps the upstream layout (TP2, full recompute, the whole optimizer on the CPU).
The rollout's MXFP8 MoE runs on vLLM's Triton kernel (`TRITON_MXFP8`); to run it on AITER's MXFP8 MoE instead,
export `VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_MOE=1` before running the script (see
[qat_rocm.md](qat_rocm.md#differences-from-nvidia-gpus)).

### Keeping every MXFP8 GEMM on HipKittens

On gfx950, Transformer Engine runs an MXFP8 GEMM on HipKittens when its token and feature dimensions are multiples of
256 and K is a multiple of 128. A GEMM that misses this falls back to hipBLASLt, whose prebuilt gfx950 MXFP8 kernels
cover only the TN layout with pre-swizzled scales; for the packed scales Transformer Engine passes, rocRoller generates
the kernel on the CPU at run time, minutes per kernel and per process (on Qwen3-8B the first `update_actor` did not
finish in 30 minutes). With `PRECISION=mxfp8` the script sets:

| Setting | Why |
|---|---|
| `expert_tensor_parallel_size=1` (`ACTOR_ETP`) | With the default ETP = TP = 2 each rank holds 768 / 2 = 384 columns of every expert, so the expert dgrad and wgrad fall back whatever the token padding |
| `moe_router_padding_for_fp8=True` with `docker/rocm/patches/megatron_mxfp8_align.py` | Pads each expert's tokens to 256 instead of Megatron's 32; the script stops if the installed Megatron is not patched |
| `actor_rollout_ref.actor.megatron.pad_to_length=True` (`PAD_TO_LENGTH_BUCKET`, 512) | Pads every packed micro-batch for the dense GEMMs; verl alone pads it to 128 |
| `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1` | Runs the experts on HipKittens' grouped MXFP8 GEMM, one launch per GEMM instead of one per expert |

Qwen3-30B-A3B's widths are multiples of 256 at TP 1 or 2 with ETP 1: hidden 2048, QKV 5120 / 2560, attention output
4096 / 2048, expert gate/up 1536 and down 768. A torch-profiler trace of `update_actor` shows only
`te_kittens::cdna4::mxfp8::mxfp8_gemm_{tn,nn,nt}_kernel` and the grouped `mxfp8_wgrad_nt_kernel` for the MXFP8 GEMMs,
and with `ROCROLLER_SAVE_ASSEMBLY=1` (rocRoller writes a `.s` file per generated kernel) no kernel is generated.

### Parallelism on MI350X

The upstream layout fits 4×B200; an MI350X has 252 GiB, so the optimizer can stay on the GPU and less needs to be
recomputed. MXFP8, 8×MI350X, rollout TP2, step 2, each row adding one change to the one above:

| Layout | step (s) | `update_actor` (s) | `old_log_prob` (s) | ref (s) | throughput (tokens/s/GPU) | peak memory (GB) |
|---|---|---|---|---|---|---|
| TP2 × EP8, full recompute, optimizer on the CPU (upstream layout) | 661 | 306 | 84 | 96 | 659 | 56 |
| + optimizer on the GPU (`OPTIMIZER_CPU_OFFLOAD=False`) | 620 | 286 | 77 | 86 | 682 | 97 |
| + TP1 (`ACTOR_TP=1`) | 452 | 164 | 44 | 69 | 941 | 123 |
| + selective recompute of `core_attn` and `moe_act` (`RECOMPUTE=selective`) | 422 | 132 | 44 | 66 | 1005 | 207 |

Peak memory is `actor/perf/max_memory_reserved_gb`. The first row ran without `SEED`, so its batch differs; it had
3.49M tokens against 3.39M for the others. Communication kernels took most of the GPU time of `update_actor` in the
profile of the upstream layout; TP1 removes the per-layer tensor-parallel part of it. Two further changes did not
help: a larger token budget per GPU failed in `update_actor` (49,152 tokens: out of memory; 32,768: an NCCL error,
likely the same cause), and rollout TP1 (eight vLLM replicas instead of four) made generation slower (157 s against
144 s).

### Results

**Configuration**

- One 8×MI350X machine for every run, `SEED=42`, so steps pair across runs; 10 steps
- TP1 × EP8 × ETP1, selective recompute, the optimizer on the GPU; rollout vLLM TP2, `gpu_memory_utilization=0.4`
- The MoE rollout on vLLM's Triton kernels, or on AITER (`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`, the
  AITER columns)
- DAPO-Math-17k deduplicated to its 17,398 unique prompts
- W&B: [wei-cai/qwen3-30b-a3b_dapo_mi350](https://wandb.ai/wei-cai/qwen3-30b-a3b_dapo_mi350), runs `bf16`, `mxfp8`,
  `bf16_aiter` and `mxfp8_aiter`

"X/Y" is training precision X, rollout precision Y. Step 1 is left out of the means: it includes one-time kernel
compilation and is much slower for BF16 (894 s) than for MXFP8 (636 s); over steps 1–10 the step time is 433 s for
BF16 and 419 s for MXFP8.

| mean of steps 2–10 | BF16/BF16 | MXFP8/MXFP8 | BF16/BF16, AITER | MXFP8/MXFP8, AITER |
|---|---|---|---|---|
| step time (s) | 381 | 395 | 383 | 395 |
| generation (s) | 141 | 141 | 134 | 134 |
| `old_log_prob` (s) | 48 | 43 | 50 | 43 |
| ref (s) | 45 | 60 | 46 | 61 |
| `update_actor` (s) | 137 | 121 | 145 | 124 |
| `update_weights` (s) | 7.6 | 28.9 | 7.9 | 32.9 |
| throughput (tokens/s/GPU) | 1078 | 1044 | 1074 | 1036 |
| `rollout_corr/kl` | 0.0018 | 0.0047 | 0.0018 | 0.0048 |
| score | −0.085 | −0.109 | −0.092 | −0.086 |
| peak memory (GB) | 208 | 210 | 208 | 205 |

Results and observations:

- With this layout MXFP8 is not faster end to end on MI350X: the step takes 3.5% longer than BF16. MXFP8 shortens
  `update_actor` by 12% and `old_log_prob` by 11%, but generation does not gain (141 s in both, against −24% on B200),
  and two costs outweigh the savings:
  - `update_weights` grows by 21 s: verl quantizes the synced weights to MXFP8 on the trainer side every step.
  - The BF16 reference forward grows by 14 s: with `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1`, which the MXFP8 actor
    needs, the reference model's BF16 experts also run Transformer Engine's grouped GEMM path (CK's `ck_tile`
    kernel in the profile) instead of one hipBLASLt GEMM per expert.
- The rollout–trainer mismatch is 2.6× BF16's (0.0047 against 0.0018), the same ratio as on B200 (0.0045 against
  0.0018).
- AITER shortens generation by 5% for both precisions (141 to 134 s) and leaves the mismatch unchanged; the step time
  does not change beyond run-to-run noise. AITER's FlyDSL MoE ran with its default configurations: it has no tuned
  configuration for these shapes.
- Ten steps are enough for speed and mismatch, not for the score.

## Example: Qwen3.5-35B-A3B

Qwen3.5-35B-A3B（MoE，40 层，GatedDeltaNet 与 full attention 按 3:1 排布，256 个 routed experts）由于下面的问题，
目前无法在 gfx950 上使用 Transformer Engine 的 MXFP8 训练 recipe。

### 问题：gfx950 上 MXFP8 GEMM 要求 K % 128 == 0

在 gfx950 上，Transformer Engine 的每一个 MXFP8 GEMM 都要求归约维度 K 是 128 的倍数；NVIDIA GPU（以及 AMD
gfx1250）只要求 32 的倍数。该检查位于 ROCm Transformer Engine 的 `cublas_gemm()` 中
（[`rocm_gemm.cu` L2223-L2234](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/rocm_gemm.cu#L2223-L2234)），在 GEMM 分派之前执行，因此稠密层和专家的 grouped GEMM
都会受到限制：

```cpp
// TODO: Also use 32 for gfx950 once hipBLASLt (and TE) support MXFP8 GEMM with
// swizzled scales on that architecture.
const int required_k_multiple = is_gfx1250 ? 32 : 128;
NVTE_CHECK((k % required_k_multiple) == 0,
           "GEMM K dimension must be multiple of ", required_k_multiple,
           " for MXFP8 scaling (got K=", k, ")");
```

Transformer Engine 以打包的、非 swizzled 的布局把 MXFP8 scale 交给 hipBLASLt
（[`HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/rocm_gemm.cu#L1660-L1671)），而 gfx950 的 kernel 沿 K 方向每次处理 128 个元素
（即 4 个 E8M0 scale）；TE 自带的 HipKittens MXFP8 kernel 同样使用 [`BLOCK_K = 128`](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/kittens/cdna4/mxfp8_gemm.cpp#L20)
（[`K % BLOCK_K == 0 && K >= 256`](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/kittens/cdna4/mxfp8_gemm.cpp#L1245)）。截至 2026/11/12，`dev` 分支上的这项检查没有变化
（[`rocm_gemm.cu` L2245-L2255 @ `027a8013`](https://github.com/ROCm/TransformerEngine/blob/027a8013f53d8f1b5528b93b31eeb7aac48c1641/transformer_engine/common/gemm/rocm_gemm.cu#L2245-L2255)）；
[ROCm/TransformerEngine#627](https://github.com/ROCm/TransformerEngine/pull/627) 只对 gfx1250 放宽到了 32。

报错信息：

```
RuntimeError: .../transformer_engine/common/gemm/rocm_gemm.hip:2233 in function cublas_gemm:
Assertion failed: (k % required_k_multiple) == 0. GEMM K dimension must be multiple of 128 for MXFP8 scaling (got K=6176)
```

对于线性层 `Y = X Wᵀ`（`W` 的形状为 `[out, in]`，`T` 为打包后的 token 数），fprop 和 dgrad 的 K 是模型宽度，
wgrad 的 K 是 token 数：

| GEMM | 计算 | K |
|---|---|---|
| fprop | `Y[T, out] = X[T, in] · Wᵀ` | `in` |
| dgrad | `dX[T, in] = dY[T, out] · W` | `out` |
| wgrad | `dW[out, in] = dYᵀ[out, T] · X[T, in]` | `T` |

Qwen3.5-35B-A3B 在两处触发了该检查（8x MI350X，TP=2，EP=8）：

| GEMM | Transformer Engine 模块 | K | 实际报错 |
|---|---|---|---|
| Routed expert 的 wgrad | [`grouped_linear.py`](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/pytorch/module/grouped_linear.py)（`grouped_gemm_wgrad`） | 路由到单个专家的 token 数 | `got K=1312`（41 x 32） |
| GatedDeltaNet `in_proj` 的 dgrad | [`layernorm_linear.py` L919-L932](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/pytorch/module/layernorm_linear.py#L919-L932) | 每个 TP rank 上 `in_proj` 的输出宽度 | `got K=6176` |

第一处的 K 是路由到单个专家的 token 数。Megatron 在 MXFP8 下按 Blackwell 的要求只把每个专家的 token 数 pad 到
32 的倍数（`megatron/core/fp8_utils.py` 中的 `get_fp8_align_size`），所以像 1312 这样是 32 的倍数但不是 128 的倍数
的值就会失败。稠密层 wgrad 的 K 是 verl 打包后的 THD 总 token 数，不受这项检查影响：verl 把每条序列 pad 到 32 的
倍数，同时把总长度补到 `TP × CP × f × 128` 的倍数（CP > 1 且为 zigzag 布局时 f = 2，否则 f = 1）
（[`util.py` L71-L80](https://github.com/verl-project/verl/blob/4b396702074369ae977db99387bd8470f02f62b9/verl/models/mcore/util.py#L71-L80)、[L316-L320](https://github.com/verl-project/verl/blob/4b396702074369ae977db99387bd8470f02f62b9/verl/models/mcore/util.py#L316-L320)），因此总 token 数始终是 128 的倍数。

第二处的 K 是权重本身的维度。Megatron 把 GatedDeltaNet 的 q、k、v、z、beta、alpha 六个投影融合成一个 `in_proj`
（`megatron/core/ssm/gated_delta_net/gdn.py` 中的 `in_proj_split_sections`）；按 Qwen3.5 的 16 个 key head、
32 个 value head、head dim 128 计算：

| 部分 | 全局宽度 | TP=2 时每个 rank |
|---|---|---|
| q | 2048 | 1024 |
| k | 2048 | 1024 |
| v | 4096 | 2048 |
| z（gate） | 4096 | 2048 |
| beta | 32 | 16 |
| alpha | 32 | 16 |
| **合计** | **12352** | **6176**（= 48 x 128 + 32） |

q/k/v/z 部分（12288 / TP）始终是 128 的倍数；beta 和 alpha 额外增加 64 / TP，因此任何 TP 大小都无法让该宽度成为
128 的倍数（TP=1 时为 12352，12352 % 128 = 64）。token padding 改变不了它；`in_proj` 是稠密的
`TELayerNormColumnParallelLinear`，更换 grouped GEMM 后端也不会改变它。该宽度是 32 的倍数，所以同一模型在 NVIDIA
GPU 上可以通过这项检查。

参考代码（ROCm Transformer Engine `023639b5`，即 `rocm/primus:v26.9` 中的版本）：

- [`rocm_gemm.cu` L2223-L2234](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/rocm_gemm.cu#L2223-L2234)：`cublas_gemm()` 中 MXFP8 的 K / M / N 检查
- [`rocm_gemm.cu` L1660-L1671](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/rocm_gemm.cu#L1660-L1671)：MXFP8 使用打包的 `VEC32_UE8M0` hipBLASLt scale 模式
- [`rocm_gemm.cu` L2362-L2390](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/rocm_gemm.cu#L2362-L2390)：MXFP8 优先分派到 HipKittens，不满足条件时回退到 hipBLASLt
- [`kittens/cdna4/mxfp8_gemm.cpp`](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/kittens/cdna4/mxfp8_gemm.cpp)：`BLOCK_K`（[L20](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/kittens/cdna4/mxfp8_gemm.cpp#L20)）及 K 的约束（[L1245](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/kittens/cdna4/mxfp8_gemm.cpp#L1245)、
  grouped [L1358-L1385](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/kittens/cdna4/mxfp8_gemm.cpp#L1358-L1385)、grouped wgrad [L1799-L1830](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/common/gemm/kittens/cdna4/mxfp8_gemm.cpp#L1799-L1830)）
- [`pytorch/module/layernorm_linear.py` L919-L932](https://github.com/ROCm/TransformerEngine/blob/023639b5168152a339e99d2e06c495dba26cf6cb/transformer_engine/pytorch/module/layernorm_linear.py#L919-L932)：GatedDeltaNet `in_proj` 报错所在的
  dgrad GEMM
- [ROCm/TransformerEngine#627](https://github.com/ROCm/TransformerEngine/pull/627)（gfx1250：K % 32）、
  [#566](https://github.com/ROCm/TransformerEngine/pull/566)（HipKittens MXFP8 GEMM，加入了 gfx950 的 TODO）
