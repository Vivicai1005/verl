# MXFP8 RL on AMD ROCm：性能分析与优化记录

Last updated: 11/26/2026

本文是 [MXFP8 RL on AMD ROCm](mxfp8_rocm.md) 的补充。下面按优化点逐条记录，每一条依次写：遇到的问题、原因、做法、对应的实验
数据和结论。如无特别说明，任务都是 Qwen3-30B-A3B GRPO（DAPO-Math-17k，64 prompts × 8 responses，最长 1024 + 8192
token），机器是一台 4×MI355X，使用 `docker/rocm/Dockerfile.rocm103` 镜像及其三个 vLLM 0.34 patch，示例脚本的默认配置，
`SEED=42`；RL 数据取第 2–10 步的均值（第 1 步含一次性的加载和编译）。

## 总览

| # | 优化点 | 作用的阶段 | 效果 | 状态 |
|---|---|---|---|---|
| 1 | 让所有 MXFP8 GEMM 都走 HipKittens | 训练 | 避免 rocRoller 现场生成 kernel（每个形状几分钟） | 已采用 |
| 2 | 训练并行布局 | 训练 | 8×MI350X 上 step 661 → 422 s | 已采用 |
| 3 | ref 显式写全 MXFP8 设置 | ref 前向 | 避免 ref 的 override 覆盖掉继承来的 MXFP8 设置，与 upstream recipe 一致 | 已采用 |
| 4 | BF16 专家改用 CK grouped GEMM | BF16 基线 | 专家 GEMM 约快 2 倍 | 已采用 |
| 5 | rollout 使用 AITER | 生成 | 生成快约 5% | 已采用 |
| 6 | vLLM MXFP8 dense linear 改为不透明算子 | 生成 | MXFP8 生成快 4.4% | 已采用（vLLM patch） |
| 7 | 升级到 ROCm 10.3 / vLLM 0.34 的必要修复 | rollout | 修复 sleep 失败和 MXFP8 refit 报错 | 已采用（vLLM patch） |

当前结果（4×MI355X，`bf16ck_v2610_aiter2` / `mxfp8_v2610_aiter_opq3`）：

| 第 2–10 步均值 | BF16/BF16 | MXFP8/MXFP8 | 差值 |
|---|---|---|---|
| step (s) | 354.7 | 353.3 | −0.4% |
| 生成 (s) | 156.7 | 143.4 | −8.5% |
| `old_log_prob` (s) | 39.5 | 40.3 | +2.2% |
| ref (s) | 39.3 | 39.9 | +1.6% |
| `update_actor` (s) | 116.4 | 123.9 | +6.5% |
| `update_weights` (s) | 2.4 | 5.3 | +2.9 s |
| 吞吐 (tokens/s/GPU) | 2309 | 2321 | +0.5% |
| `rollout_corr/kl` | 0.0018 | 0.0049 | |

MXFP8 端到端和 BF16 打平：生成阶段快 8.5%，被训练侧（`update_actor`、`update_weights`）抵消。剩下的开销见
[剩余开销与后续方向](#剩余开销与后续方向)。

## 1. 让所有 MXFP8 GEMM 都走 HipKittens

**问题**：在 gfx950 上，Transformer Engine 只有当 GEMM 的 token 维和特征维都是 256 的倍数、K 是 128 的倍数时才走
HipKittens。不满足时退回 hipBLASLt，而 hipBLASLt 在 gfx950 上预编译的 MXFP8 kernel 只覆盖 TN 布局和预先 swizzle 过的
scale；对 TE 传入的打包 scale，rocRoller 会在运行时用 CPU 现场生成 kernel，每个形状、每个进程几分钟。

**原因**：三处形状不满足要求：

- Megatron 按 NVIDIA 的要求，只把每个专家的 token 数补齐到 32；
- 默认 ETP = TP = 2 时，每个 rank 只持有每个专家的 768 / 2 = 384 列；
- verl 只把打包后的序列补齐到 128。

**做法**：

| 设置 | 作用 |
|---|---|
| verl 在 ROCm 上把 MXFP8 的专家 token 对齐从 32 提到 256（`apply_patch_rocm_mxfp8_expert_alignment`，配合 `moe_router_padding_for_fp8=True`） | 专家 GEMM 的 token 维满足 256 |
| `expert_tensor_parallel_size=1`（`ACTOR_ETP`） | 专家宽度保持 1536 / 768 |
| `actor_rollout_ref.actor.megatron.pad_to_length=True`，`PAD_TO_LENGTH_BUCKET=512` | dense GEMM 的 token 维满足 256 |
| `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1` | 专家走 HipKittens 的 grouped MXFP8 GEMM，每个 GEMM 一次 launch |

在 TP 1 或 2、ETP 1 时，Qwen3-30B-A3B 的各个宽度都是 256 的倍数：hidden 2048，QKV 5120 / 2560，attention 输出
4096 / 2048，专家 gate/up 1536、down 768。

**实验数据**：

- `update_actor` 的 profiler trace 里，MXFP8 GEMM 只有 `te_kittens::cdna4::mxfp8::mxfp8_gemm_{tn,nn,nt}_kernel`
  和 grouped 的 `mxfp8_wgrad_nt_kernel`；
- 设置 `ROCROLLER_SAVE_ASSEMBLY=1`（rocRoller 每生成一个 kernel 写一个 `.s` 文件）后，所有 MXFP8 实验都是 0 个 `.s` 文件；
- 用不打补丁的原版 Megatron、只靠 verl 侧对齐跑 2 步 MXFP8，同样没有生成 kernel，第 2 步 367 s，和打补丁时一致。

**结论**：这几项是 MXFP8 在 gfx950 上能正常训练的前提。专家 token 补齐到 256 平均每个专家多补约 128 个空 token，相对每个
专家每个 micro-batch 上千个 token 是 10% 以内的额外计算，和退回 rocRoller 相比可以忽略。

## 2. 训练并行布局

**问题**：upstream 的布局（TP2 × EP、full recompute、优化器全放 CPU）是为 4×B200 准备的，在 MI350 / MI355 上不是最优。

**做法与实验数据**：在 8×MI350X（每卡 252 GiB，`rocm/primus:v26.9`，vLLM 0.32）上逐项改动，MXFP8，rollout TP2，
取第 2 步，每一行在上一行的基础上只改一处：

| 布局 | step (s) | `update_actor` (s) | `old_log_prob` (s) | ref (s) | 吞吐 (tokens/s/GPU) | 显存峰值 (GB) |
|---|---|---|---|---|---|---|
| TP2 × EP8，full recompute，优化器在 CPU（upstream 布局） | 661 | 306 | 84 | 96 | 659 | 56 |
| + 优化器放到 GPU（`OPTIMIZER_CPU_OFFLOAD=False`） | 620 | 286 | 77 | 86 | 682 | 97 |
| + TP1（`ACTOR_TP=1`） | 452 | 164 | 44 | 69 | 941 | 123 |
| + 对 `core_attn` 和 `moe_act` 做 selective recompute（`RECOMPUTE=selective`） | 422 | 132 | 44 | 66 | 1005 | 207 |

- **TP1 收益最大：** upstream 布局的 profile 里，通信 kernel 占了 `update_actor` 大部分 GPU 时间；TP1 去掉了每层的
  tensor-parallel 通信。
- **没有帮助的改动：** 加大每卡 token 预算会让 `update_actor` 失败（49,152 显存不足；32,768 报 NCCL 错误，很可能同因）；
  rollout 改成 TP1（8 个 vLLM 副本）生成反而变慢（157 s 对 144 s）。
- **4×MI355X 上：** EP4 时每张卡放四分之一的专家及其 Adam 状态，优化器放在 GPU 时第 2 步 `update_actor` 反向显存不足
  （已分配 256 GiB / 288 GiB），所以优化器仍放在 CPU。

**结论**：示例默认 TP1 × EP4 × ETP1、selective recompute、优化器在 CPU。8×MI350X 上 MXFP8 的 step 从 661 s 降到 422 s。

## 3. ref 显式写全 MXFP8 设置

**问题**：verl 的 `megatron_ref.yaml` 让 ref 默认继承 actor 的 `override_transformer_config`，所以 upstream 脚本只给
actor 设置 MXFP8，ref 也自动以 MXFP8 运行。但在 Hydra 里，只要给 ref 设置了任何一个
`++actor_rollout_ref.ref.megatron.override_transformer_config.<key>`，继承来的整个字典就会被替换成只含这些项的新字典。
ROCm 脚本必须给 ref 设置 `attention_backend=fused`（镜像导出了 `NVTE_FLASH_ATTN=0` / `NVTE_FUSED_ATTN=1`，Megatron 会
检查两者一致），这一行把 ref 本该继承的 fp8 / mxfp8 和 MoE 设置一并覆盖掉了，ref 变成以 BF16 运行。

**做法**：既然 ref 必须单独设置 `attention_backend`，就把 ref 前向用到的设置全部显式写出来：

```bash
REF_TF=actor_rollout_ref.ref.megatron.override_transformer_config
REF=(
    # ...
    # 设置 ref 的任何 override 都会替换掉从 actor 继承的字典，所以 ref 前向需要的设置都要在这里写全
    ++${REF_TF}.moe_grouped_gemm=True
    ++${REF_TF}.moe_token_dispatcher_type=alltoall
    ++${REF_TF}.moe_router_dtype=fp32
    ++${REF_TF}.attention_backend=fused
)

# PRECISION=mxfp8 时追加到 PRECISION_ARGS，ref 与 actor 一样以 MXFP8 运行
++${REF_TF}.fp8=e4m3
++${REF_TF}.fp8_recipe=mxfp8
++${REF_TF}.moe_router_padding_for_fp8=True
```

ref 只做前向，不需要 recompute 相关的设置；`pad_to_length` 不在 `override_transformer_config` 里，ref 仍从 actor 继承。

**验证**：用 Hydra compose 检查最终配置：`PRECISION=mxfp8` 时，ref 的 `override_transformer_config` 与 actor 相同（除
recompute 相关项外），包含 fp8=e4m3、fp8_recipe=mxfp8、`moe_router_padding_for_fp8` 和上面的 MoE 设置；`PRECISION=bf16`
时只有 MoE 设置和 `attention_backend=fused`。训练日志打印的配置里，`fp8_recipe: mxfp8` 同时出现在 actor 和 ref 下。

**结论**：ROCm 脚本与 upstream recipe 一致，ref 和 actor 一样以 MXFP8 运行。以后给 ref 增加任何 override 时，都要连同
它依赖的其余设置一起写全。

## 4. BF16 专家改用 CK grouped GEMM

**问题**：BF16 基线的专家 GEMM 一直是每个专家单独一次 hipBLASLt GEMM，32 个本地专家时一次前向加反向约 96 次 launch。
脚本原来设置的 `NVTE_USE_GROUPED_GEMM_TRITON=1` 实际没有生效。

**原因**：Transformer Engine 对 BF16 `GroupedLinear` 有三条路径，每次调用按环境变量选择：

| 路径 | 怎么选中 | 说明 |
|---|---|---|
| 每个专家一次 hipBLASLt GEMM | 默认 | 约 96 次 launch |
| CK grouped GEMM（`ck_tile::GroupedGemmKernel`） | `NVTE_USE_CK_GROUPED_GEMM=1`；设置 `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1` 时 BF16 也走这里 | fprop、dgrad、wgrad 各一次 launch |
| Triton grouped GEMM | `NVTE_USE_GROUPED_GEMM_TRITON=1` | 只在非 FP8 且 `fuse_wgrad_accumulation` 关闭时生效；Megatron 会打开它，所以不会走 |

而旧版 TE（`023639b5`，`rocm/primus:v26.9`）的 CK 路径在 N 是 256 倍数时用 256×256×64 tile，在 ROCm ≥ 7.14 编译器下会
溢出寄存器，反而更慢。

**做法**：升级到包含 [ROCm/TransformerEngine#783](https://github.com/ROCm/TransformerEngine/pull/783) 的
`rocm/primus:v26.10`（TE `b5c00f432`，换掉了这些 tile），BF16 分支改设 `NVTE_USE_CK_GROUPED_GEMM=1`。

**实验数据**（32 个本地专家，20,370 个 token 不均匀分配，`fuse_wgrad_accumulation=True`，一次前向加反向）：

| 形状 | hipBLASLt 逐专家 | 旧 CK（v26.9，只计 GEMM kernel） | 新 CK（v26.10） |
|---|---|---|---|
| FC1 2048 → 1536 | 1.88 ms | 3.61 ms（同口径 hipBLASLt 1.51 ms） | 0.94 ms |
| FC2 768 → 2048 | 1.25 ms | — | 0.58 ms |

CK 与 hipBLASLt 的相对误差最大 3.7e-5（fprop、dgrad、wgrad）。在 RL 里，强制走 Triton 路径（配合
`gradient_accumulation_fusion=False`）的第 2–4 步 step 为 374 / 366 / 355 s，hipBLASLt 基线为 370 / 383 / 358 s，没有收益。

**结论**：BF16 基线的专家 GEMM 约快 2 倍，BF16 基线变得更强，MXFP8 和 BF16 的对比因此更公平。

## 5. rollout 使用 AITER

**问题**：vLLM 默认的 MoE 用 Triton kernel（BF16 的 `fused_moe` 在 MI355 上没有调优配置；MXFP8 用 `TRITON_MXFP8`）。

**做法**：两种精度都设置 `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`。AITER 的 MXFP8 MoE 用 FlyDSL kernel。需要配合
两处修复：vLLM 0.32 里 AITER 的 MXFP8 MoE 不接受 SiLU（vLLM 0.33 起 upstream 已支持）；verl 在 refit 重新整理 FP8 参数时
会丢掉 AITER 设置的 `is_shuffled` 标记，导致 refit 后 MXFP8 MoE 出错（已在 verl 中修复）。

**实验数据**：

| | Triton MoE | AITER | 变化 |
|---|---|---|---|
| decode step，BF16，256 条序列，vLLM 0.32（ms） | 17.19 | 15.48 | −10% |
| decode step，MXFP8，256 条序列，vLLM 0.32（ms） | 15.24 | 14.18 | −7% |
| RL 生成（8×MI350X，两种精度相同，s） | 141 | 134 | −5% |

decode 中，AITER 的 MXFP8 MoE GEMM 为 3.1 ms，BF16 的 CK MoE 为 5.4 ms；AITER 还把 allreduce 和 RMSNorm 融合在一起。
`rollout_corr/kl` 不变。

**结论**：两种精度的生成都快约 5%，示例默认推荐开启。

## 6. vLLM MXFP8 dense linear 改为不透明算子

**问题**：decode profile 里，MXFP8 的 dense GEMM（QKV 和输出投影）比 BF16 还慢：每步 2.16 ms 对 1.72 ms，每次调用约
18 µs；而单独的 kernel 微基准里，同样的形状只要约 9–10 µs。

**原因**：vLLM 的 gfx950 MXFP8 linear 按 token 数 M 选择 tile（`rocm_native.py` 的 `_select_cfg`）。这个调用位于经过
torch.compile 的前向里，tile 只在 vLLM 编译时的那次 profile run 里选一次（M = `max_num_batched_tokens` = 8192），之后
所有 CUDA graph 都沿用这个为预填充准备的 tile。256 条序列 decode 时，QKV 和输出投影用的是 128×128×256 tile，grid 只有
2×20 和 2×16，即 32–40 个 workgroup，对应 GPU 的 256 个计算单元。eager 模式下 M = 256 会选 64×64×512（grid 4×40 /
4×32，约 10 µs）。同文件里的 block32 路径已经因为同样的原因放在了不透明算子后面，逐行 scale 路径没有。

**做法**：`vllm-0.34-rocm-mxfp8-linear-opaque-op.patch` 把逐行 scale 路径注册成 `torch.ops.vllm.rocm_mxfp8_linear` /
`rocm_mxfp8_gemm`，每张 CUDA graph 捕获时按自己的 M 选 tile。输出与原函数逐位一致（M = 8192、256、16、1、0，eager 和
torch.compile 下都验证过）。

**实验数据**：decode step（TP2，256 条序列，AITER），rank 0 每步的 GPU kernel 时间（ms）：

| | BF16 | MXFP8，不带 patch | MXFP8，带 patch |
|---|---|---|---|
| decode step，不开 profiler | 15.06 | 13.85 | 13.04 |
| dense GEMM（QKV、输出、router、`lm_head`） | 1.72 | 2.16 | 1.35 |
| QKV / 输出投影每次调用 | — | 17.9 / 18.1 µs | 9.7 / 9.2 µs |
| MoE GEMM | 5.75 | 3.15 | 3.15 |

RL（同四张卡依次运行，`mxfp8_v2610_aiter_base2` 与 `mxfp8_v2610_aiter_opq3`）：

| 第 2–10 步均值 | BF16/BF16 | MXFP8，不带 patch | MXFP8，带 patch |
|---|---|---|---|
| step (s) | 354.7 | 360.9 | 353.3 |
| 生成 (s) | 156.7 | 150.1 | 143.4 |
| 吞吐 (tokens/s/GPU) | 2309 | 2265 | 2321 |
| `rollout_corr/kl` | 0.0018 | 0.0049 | 0.0049 |

**结论**：MXFP8 生成快 4.4%，十步里每一步都更快，不匹配度不变；MXFP8 由每步比 BF16 慢 1.7% 变为快 0.4%。RL 里的收益
小于满载 decode 的 5.8%，因为预填充不受影响，而且序列陆续结束后 batch 变小。这个问题在普通 MXFP8 推理里同样存在，值得提给
vLLM upstream。

## 7. 升级到 ROCm 10.3 / vLLM 0.34 的必要修复

**问题**：为了拿到第 4 节的 CK 修复和更新的 vLLM，镜像从 `rocm/primus:v26.9` + vLLM 0.32 升级到 `rocm/primus:v26.10`
（ROCm 10.3、TE `b5c00f432`）+ vLLM 0.34。升级后遇到两个阻塞问题。

**原因与做法**：

| 问题 | 原因 | 修复 |
|---|---|---|
| vLLM 第一次 sleep 就失败（`cumem_allocator.cpp:274 invalid argument`），wake 时显存不足，一步都跑不了 | vLLM 0.34 把 sleep 时的 CPU 备份改成 mmap + `cudaHostRegister`；在 ROCm 上抓取 HIP graph 后，部分权重分配的 `hipMemUnmap` 会失败 | `vllm-0.34-rocm-cumem-pinned-backup.patch`：ROCm 上改回 torch pinned tensor（vLLM 0.32 的写法） |
| MXFP8 第一次 refit 报 scale 形状不匹配 | verl 用 dummy 权重启动 vLLM，scale 全部相同，被误判成 32×32 块格式并压缩 | `vllm-0.34-rocm-mxfp8-block32-scales-switch.patch`：加 `VLLM_ROCM_MXFP8_BLOCK32_SCALES` 开关，设为 0 |

另外 vLLM 0.34 要求 `huggingface_hub >= 2`，transformers 从 5.18 起才允许，镜像固定 transformers 5.18.0。

**实验数据**（cumem 问题的对照实验，独立复现脚本，Qwen3-30B-A3B，TP2）：

| 环境 | HIP graph | CPU 备份方式 | 结果 |
|---|---|---|---|
| v26.10 + vLLM 0.34，BF16 / MXFP8 | 开 | mmap + `cudaHostRegister`（0.34 原样） | 失败：每个 rank 8–14 个权重分配 unmap 失败 |
| v26.10 + vLLM 0.34，MXFP8 | 关 | 0.34 原样 | 通过 |
| v26.10 + vLLM 0.34，BF16 | 开 | mmap，不调用 `cudaHostRegister` | 通过 |
| v26.10 + vLLM 0.34，BF16 / MXFP8 | 开 | torch pinned tensor（修复后） | 通过（BF16 测 sleep level 1；MXFP8 测 level 1 和 2） |
| v26.9 + vLLM 0.32，BF16 | 开 | torch pinned tensor | 通过 |

**结论**：两个 patch 都是升级后能跑起来的前提，与精度设置无关。HIP 运行时内部的具体原因还没查清，详见内部记录
`docs/low_precision/rocm_issues/primus-v26.10/`。

## 尝试过但没有采用的方向

| 方向 | 结果 | 原因 |
|---|---|---|
| dense MXFP8 GEMM 换成 aiter FlyDSL（`gemm_a8w8_mxfp8_bpreshuffle`） | 256 条序列时 QKV / 输出 11.6 / 8.8 µs，vLLM 原 kernel 只算 GEMM 为 9.4 / 8.8 µs；M = 4096 时 54 对 42 µs | 没有一致的收益。真正的问题是第 6 节的 tile 冻结：这组微基准在 eager 下运行，看不到它 |
| dense MXFP8 GEMM 换成 hipBLASLt MX（torch `scaled_mm`） | M < 64 报错（scale 需要补齐），误差是其他实现的 2 倍（3.3e-3 对 1.66e-3），没有速度优势 | 不可用 |
| 把激活量化融合进 RMSNorm（aiter `fused_rms_mxfp8_quant`） | 256 条序列时 2.0 µs 对分开做的 5.8 µs | 与 vLLM 的量化器不是逐位一致：amax 恰为 448 × 2^k 的 block 会选大一档的 scale，RMSNorm 舍入也不同，会改变 rollout 数值 |
| BF16 专家走 Triton grouped GEMM | 第 2–4 步与 hipBLASLt 持平（见第 4 节） | 无收益 |
| rollout TP1（8 个 vLLM 副本） | 生成 157 s 对 144 s | 更慢 |
| 每卡 token 预算 32,768 / 49,152 | `update_actor` 失败 | 显存不足 |
| dense 层在 rollout 保持 BF16、FP8 KV cache | 未测 | 与 B200 recipe 的精度设置不一致，不作为加速手段 |

## 剩余开销与后续方向

MXFP8 在生成阶段已经领先，端到端打平是因为训练侧和权重同步还有额外开销。

**训练侧逐专家量化。** HipKittens 的 grouped MXFP8 GEMM 本身很快，但前后的量化是逐专家进行的。`GroupedLinear` 微基准
（32 个专家，2048 → 1536，20.7k token 补齐到 256，一次前向加反向，`rocm/primus:v26.9`）：

| | BF16，hipBLASLt 逐专家 | MXFP8，HipKittens grouped |
|---|---|---|
| GEMM kernel | 1.52 ms（约 96 次 launch） | 0.49 ms（3 次 launch） |
| 量化和布局 kernel | — | 约 0.85 ms：`_cast_transpose_triton_mxfp8` ×32、`pack_scales_kernel` ×64、`multi_quantize` ×2，以及 fill 和 copy |
| 合计 | 1.52 ms | 1.34 ms |

GEMM 约 3 倍的收益大部分被逐专家的量化和 scale 打包吃掉；BF16 专家又换成了更快的 CK grouped GEMM，所以 MXFP8 的
`update_actor` 现在比 BF16 慢 6.5%。把这些逐专家的 kernel 合并成 grouped 的版本，是训练侧最主要的方向。

**权重同步。** `update_weights` MXFP8 为 5.3 s，BF16 为 2.4 s；rollout 一侧测到的每步耗时为 4.4–5.7 s 对 1.5 s。
MXFP8 时，训练侧要把每个权重量化一遍（TileLang，1×32 E8M0 scale）；vLLM 先把权重和 scale 写进 checkpoint 布局的暂存
buffer，再逐层重新执行 `process_weights_after_loading`（AITER 的 MoE 重排、linear kernel 的重新打包），最后拷回 CUDA
graph 使用的显存。BF16 是原地加载。两侧各占多少还没有单独测量。

**decode 中剩余的 MXFP8 专有开销**（每步，ms，带第 6 节 patch）：

| 开销 | BF16 | MXFP8 |
|---|---|---|
| MoE 量化与排序（`aiter::fused_mx_quant_moe_sort_kernel` 等） | 0.47 | 1.13 |
| dense 激活量化 | — | 0.50 |
| copy 与 elementwise（每层多一次设备内拷贝） | 0.11 | 0.35 |

## 已知限制：Qwen3.5-35B-A3B（gfx950 上 MXFP8 GEMM 要求 K % 128 == 0）

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
