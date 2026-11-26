# MXFP8 RL on AMD ROCm：性能分析与优化记录

Last updated: 11/26/2026

本文是 [MXFP8 RL on AMD ROCm](mxfp8_rocm.md) 的补充，记录其中 ROCm 相关设置的由来、背后的对比实验，以及时间都花在了哪里。
如无特别说明，下面的数据都来自一台 4×MI355X 机器上的 Qwen3-30B-A3B，使用 `docker/rocm/Dockerfile.rocm.primus-v26.10`
镜像及其三个 vLLM 0.34 patch，采用示例脚本的默认配置，rollout 的 MoE 走 AITER。

## 让所有 MXFP8 GEMM 都走 HipKittens

在 gfx950 上，只有当 token 维和特征维都是 256 的倍数、K 是 128 的倍数时，Transformer Engine 才会把 MXFP8 GEMM 交给
HipKittens。不满足条件的 GEMM 会退回 hipBLASLt，而 hipBLASLt 在 gfx950 上预编译的 MXFP8 kernel 只覆盖 TN 布局和
预先 swizzle 过的 scale；对 Transformer Engine 传入的打包 scale，rocRoller 会在运行时用 CPU 现场生成 kernel，每个
kernel、每个进程都要花几分钟（在 Qwen3-8B 上，第一次 `update_actor` 30 分钟都没跑完）。`PRECISION=mxfp8` 时，示例脚本
设置了：

| 设置 | 原因 |
|---|---|
| `expert_tensor_parallel_size=1`（`ACTOR_ETP`） | 默认 ETP = TP = 2 时，每个 rank 只持有每个专家的 768 / 2 = 384 列，无论 token 怎么补齐，专家的 dgrad 和 wgrad 都会退回 |
| `moe_router_padding_for_fp8=True` | 把每个专家的 token 数补齐到 256，而不是 Megatron 默认的 32（verl 在 ROCm 上对 MXFP8 提高了对齐要求） |
| `actor_rollout_ref.actor.megatron.pad_to_length=True`（`PAD_TO_LENGTH_BUCKET`，512） | 为 dense GEMM 把每个打包后的 micro-batch 补齐；verl 自身只补齐到 128 |
| `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1` | 让专家走 HipKittens 的 grouped MXFP8 GEMM，每个 GEMM 一次 launch，而不是每个专家一次 |

在 TP 1 或 2、ETP 1 时，Qwen3-30B-A3B 的各个宽度都是 256 的倍数：hidden 2048，QKV 5120 / 2560，attention 输出
4096 / 2048，专家 gate/up 1536、down 768。`update_actor` 的 torch profiler trace 里，MXFP8 GEMM 只有
`te_kittens::cdna4::mxfp8::mxfp8_gemm_{tn,nn,nt}_kernel` 和 grouped 的 `mxfp8_wgrad_nt_kernel`；设置
`ROCROLLER_SAVE_ASSEMBLY=1`（rocRoller 每生成一个 kernel 就写一个 `.s` 文件）后，也没有生成任何 kernel。

## 训练并行布局

upstream 的布局是为 4×B200 准备的。ROCm 示例的默认配置（TP1 × EP4 × ETP1、selective recompute、优化器放在 CPU）来自
早先在 8×MI350X（每卡 252 GiB）上做的一组筛选，当时用的是 `rocm/primus:v26.9` 镜像和 vLLM v0.32.0。MXFP8，rollout
TP2，取第 2 步，每一行在上一行的基础上只改一处：

| 布局 | step (s) | `update_actor` (s) | `old_log_prob` (s) | ref (s) | 吞吐 (tokens/s/GPU) | 显存峰值 (GB) |
|---|---|---|---|---|---|---|
| TP2 × EP8，full recompute，优化器在 CPU（upstream 布局） | 661 | 306 | 84 | 96 | 659 | 56 |
| + 优化器放到 GPU（`OPTIMIZER_CPU_OFFLOAD=False`） | 620 | 286 | 77 | 86 | 682 | 97 |
| + TP1（`ACTOR_TP=1`） | 452 | 164 | 44 | 69 | 941 | 123 |
| + 对 `core_attn` 和 `moe_act` 做 selective recompute（`RECOMPUTE=selective`） | 422 | 132 | 44 | 66 | 1005 | 207 |

显存峰值取 `actor/perf/max_memory_reserved_gb`。第一行没有设置 `SEED`，所以那一步的 batch 不同：它有 3.49M 个 token，
其余几行是 3.39M。在 upstream 布局的 profile 里，通信 kernel 占了 `update_actor` 大部分的 GPU 时间；TP1 去掉了其中每层的
tensor-parallel 通信。另外两项改动没有帮助：加大每卡的 token 预算会让 `update_actor` 失败（49,152 个 token 时显存不足；
32,768 时报 NCCL 错误，很可能是同一原因）；rollout 改成 TP1（8 个 vLLM 副本代替 4 个）反而让生成变慢（157 s 对 144 s）。

在 4×MI355X（每卡 288 GiB）上，同样的布局改成 EP4、优化器放在 GPU 时，第 1 步能跑完，但第 2 步 `update_actor` 的反向
显存不足（已分配 256 GiB）：每张卡要放四分之一的专家以及它们的 Adam 状态。因此示例默认把整个优化器放在 CPU
（`OPTIMIZER_CPU_OFFLOAD=True`）。

## BF16 专家的 GEMM

Transformer Engine 对 BF16 的 `GroupedLinear`（也就是 MoE 专家）有三条路径，每次调用时按环境变量选择：

| 路径 | 怎么选中 | 说明 |
|---|---|---|
| 每个专家一次 hipBLASLt GEMM | 默认 | 32 个本地专家时，一次前向加反向约 96 次 launch |
| CK grouped GEMM（`ck_tile::GroupedGemmKernel`） | `NVTE_USE_CK_GROUPED_GEMM=1`；设置了 `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1` 时 BF16 也会走这里 | fprop、dgrad、wgrad 各一次 launch |
| Triton grouped GEMM | `NVTE_USE_GROUPED_GEMM_TRITON=1` | 只在非 FP8、且 `fuse_wgrad_accumulation` 关闭时生效；Megatron 会打开它，所以这条路径实际不会走 |

`rocm/primus:v26.9` 里的 Transformer Engine `023639b5` 只要 N 是 256 的倍数就用 256×256×64 的 CK tile，这个 tile 在
ROCm ≥ 7.14 的编译器下会溢出寄存器：专家 FC1（2048 → 1536，只计 GEMM kernel）上 CK 要 3.61 ms，hipBLASLt 只要 1.51 ms。
[ROCm/TransformerEngine#783](https://github.com/ROCm/TransformerEngine/pull/783) 换掉了这些 tile；`rocm/primus:v26.10`
（Transformer Engine `b5c00f432`）已经包含这个修复。在 v26.10 上，32 个本地专家、20,370 个 token 不均匀分配、
`fuse_wgrad_accumulation=True`，一次前向加反向：

| 形状 | hipBLASLt 逐专家 | CK grouped |
|---|---|---|
| FC1 2048 → 1536 | 1.88 ms | 0.94 ms |
| FC2 768 → 2048 | 1.25 ms | 0.58 ms |

CK 与 hipBLASLt 的相对误差最大 3.7e-5（fprop、dgrad、wgrad 都在内）。示例的 BF16 运行使用 CK 路径
（`NVTE_USE_CK_GROUPED_GEMM=1`）。在 RL 里强制走 Triton 路径（配合 `gradient_accumulation_fusion=False`），相比
hipBLASLt 没有收益。

## 训练中的 MXFP8 专家 GEMM

HipKittens 的 grouped MXFP8 GEMM 在每次 `GroupedLinear` 前向加反向里只需要三次 launch（fprop、dgrad 和一次 grouped
wgrad），但它前后的量化是逐专家进行的。一组 `GroupedLinear` 微基准（32 个专家，2048 → 1536，20.7k 个 token 补齐到 256，
一次前向加反向，`rocm/primus:v26.9`）：

| | BF16，hipBLASLt 逐专家 | MXFP8，HipKittens grouped |
|---|---|---|
| GEMM kernel | 1.52 ms（约 96 次 launch） | 0.49 ms（3 次 launch） |
| 量化和布局相关 kernel | — | 约 0.85 ms：`_cast_transpose_triton_mxfp8` ×32、`pack_scales_kernel` ×64、`multi_quantize` ×2，以及 fill 和 copy |
| 合计 | 1.52 ms | 1.34 ms |

逐专家的量化和 scale 打包吃掉了 GEMM 本身约 3 倍收益中的大部分；而 BF16 专家又换到了更快的 CK grouped GEMM，所以在 RL
里 MXFP8 的 `update_actor` 现在比 BF16 慢 6.5%（123.9 s 对 116.4 s）。这些逐专家的 kernel 是训练侧接下来主要的优化目标。

## rollout 中的 MXFP8 dense linear：`vllm-0.34-rocm-mxfp8-linear-opaque-op.patch`

vLLM 的 gfx950 MXFP8 linear kernel 按 token 数 M 选择 tile（`rocm_native.py` 里的 `_select_cfg`）。这个调用位于经过
torch.compile 的前向里，因此 tile 只在 vLLM 触发编译的那次 profile run 里选一次（M = `max_num_batched_tokens`，即 8192），
之后所有 CUDA graph 都沿用这个为预填充准备的 tile。在 256 条序列的 decode trace 里，QKV 和输出投影用的是 128×128×256
的 tile，grid 为 2×20 和 2×16，只有 32–40 个 workgroup 对应 GPU 的 256 个计算单元，每次调用约 18 µs。eager 模式在
M = 256 时会选 64×64×512（grid 4×40 和 4×32，约 10 µs）。同一文件里的 block32 路径正是因为这个原因放在了不透明的自定义
算子后面。这个 patch 把逐行 scale 路径注册成 `torch.ops.vllm.rocm_mxfp8_linear` / `rocm_mxfp8_gemm`，让每张 CUDA graph
按自己的 M 选 tile：256 条序列时，graph 里用的就是 64×64×512 的 tile，每次调用 9.7 和 9.2 µs。它的输出与未打 patch 的
函数逐位一致（M = 8192、256、16、1、0，eager 和 torch.compile 下都验证过）。

一个 decode step（TP2，256 条序列，AITER），rank 0 每步的 GPU kernel 时间（ms）：

| | BF16 | MXFP8，不带 patch | MXFP8，带 patch |
|---|---|---|---|
| decode step，不开 profiler (ms) | 15.06 | 13.85 | 13.04 |
| MoE GEMM | 5.75 | 3.15 | 3.15 |
| MoE 量化与排序 | 0.47 | 1.13 | 1.13 |
| dense GEMM（QKV、输出、router、`lm_head`） | 1.72 | 2.16 | 1.35 |
| dense 激活量化 | — | 0.49 | 0.50 |
| copy 与 elementwise | 0.11 | 0.35 | 0.35 |
| attention | 2.21 | 2.19 | 2.19 |
| allreduce（含融合的 RMSNorm） | 3.65 | 3.59 | 3.58 |

在 RL 里，三组都在同样的四张卡上依次运行（W&B run `bf16ck_v2610_aiter2`、`mxfp8_v2610_aiter_base2` 和
`mxfp8_v2610_aiter_opq3`），取第 2–10 步的均值：

| | BF16/BF16 | MXFP8/MXFP8，不带 patch | MXFP8/MXFP8，带 patch |
|---|---|---|---|
| step 时间 (s) | 354.7 | 360.9 | 353.3 |
| 生成 (s) | 156.7 | 150.1 | 143.4 |
| `update_actor` (s) | 116.4 | 122.4 | 123.9 |
| `update_weights` (s) | 2.4 | 5.8 | 5.3 |
| 吞吐 (tokens/s/GPU) | 2309 | 2265 | 2321 |
| `rollout_corr/kl` | 0.0018 | 0.0049 | 0.0049 |

这个 patch 让生成缩短 4.4%，十步里每一步都更快，不匹配度不变；不带它时 MXFP8 每步比 BF16 慢 1.7%，带上它快 0.4%。RL
里的收益低于满载 decode step 的 5.8%，因为预填充不受影响，而且随着序列陆续结束 batch 会变小。剩下的 MXFP8 专有 decode
开销有：MoE 的量化与排序（`aiter::fused_mx_quant_moe_sort_kernel`，每步比 BF16 多 0.66 ms）、dense 激活量化（0.50 ms），
以及每层多出的一次设备内拷贝（0.23 ms）。把激活量化融合进前面的 RMSNorm 可以去掉其中一部分，但 AITER 的 MXFP8 量化器
和 vLLM 的不是逐位一致的：对 amax 恰好等于 448 × 2^k 的 block，它会选大一档的 scale，而且 `fused_rms_mxfp8_quant` 的
RMSNorm 舍入也不同，用它会改变 rollout 的数值。

## `update_weights`

MXFP8 的 `update_weights` 要 5.3 s，BF16 是 2.4 s；在 rollout 一侧测到的每步耗时分别是 4.4–5.7 s 和 1.5 s。MXFP8 时，
训练侧要把每个同步的权重量化一遍（TileLang，1×32 E8M0 scale）；vLLM 先把权重和 scale 写进 checkpoint 布局的暂存
buffer，再逐层重新执行 `process_weights_after_loading`（AITER 的 MoE 重排、linear kernel 的重新打包），最后拷回 CUDA
graph 使用的显存。BF16 则是原地加载权重。训练侧和 rollout 侧各占多少还没有单独测量。

## 在共享机器上做测量

这里用的 4×MI355X 是一台共享机器：主机内存 3 TB，没有 swap，其他用户放在 `/dev/shm` 里的文件就占了其中 1.25–1.4 TB。
训练进程的主机内存峰值约 760 GB（CPU 上的优化器、参数 offload，以及 rollout sleep 时的 CPU 备份）。别人的任务启动时，
空闲内存可能降到 10–20 GB，内核开始直接回收内存，表现为 `update_actor`、ref、`old_log_prob` 和 `update_weights` 变慢
（有一次 `update_weights` 用了 67 s，平时 5 s），还有一次生成阶段四张卡同时停顿约 50 s。在运行时每秒记录
`/proc/meminfo`、`/proc/vmstat`（`pgscan_direct`、`allocstall_*`）和 kworker 数量，就足以把这些受干扰的步区分出来；
上面的对比都使用在同一批卡上依次运行的实验。

## Qwen3.5-35B-A3B: MXFP8 GEMM K % 128 on gfx950

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
