# rocm/primus:v26.10 镜像升级记录

Last updated: 11/26/2026

记录日期：2026-11-24　　机器：smci355-ccs-aus-n06-05（MI355X，gfx950）

镜像 `verl-qat:primus-v26.10`（Dockerfile：`docker/rocm/Dockerfile.rocm.primus-v26.10`）把 base 从 rocm/primus:v26.9 换成 v26.10，vLLM 从 0.32.0 换成 0.34.0。升级中遇到两件需要记录的事：

- **第一部分：** vLLM 0.34 在 ROCm 上 sleep 失败。已用临时 patch 绕过，根因未完全查清。
- **第二部分：** v26.10 里的 TE 修复了 CK BF16 grouped GEMM 的性能问题，BF16 专家 GEMM 因此改用 CK grouped。已验证，示例脚本已修改。

| | v26.9 镜像 | v26.10 镜像 |
|---|---|---|
| torch | 2.16.0a0+rocm10.2.0a20261020 | 2.16.0a0+rocm10.3.0a20261111 |
| HIP | 7.17.26416 | 7.18.26451 |
| Transformer Engine | 023639b5（2026-10-20） | 2.21.0.dev0+b5c00f432（2026-11-11，含 ROCm/TE #783） |
| vLLM | v0.32.0（源码编译） | v0.34.0（62d4bc3c7，源码编译） |

---

# 第一部分：vLLM 0.34 在 ROCm 上 sleep 失败（cumem_allocator.cpp:274）

## 1.1 现象

在 v26.10 镜像上跑 Qwen3-30B-A3B GRPO（MXFP8，2 步冒烟测试）。vLLM 第一次 sleep 时，所有 TP worker 都报错，训练一步都没跑就退出（EXIT=1）：

```
gpu_worker.sleep -> sleep_mode_backend.suspend -> CuMemAllocator.sleep -> _put_to_sleep -> unmap_and_release
RuntimeError: CUDA Error: invalid argument at /workspace/vllm/csrc/cumem_allocator.cpp:274
```

如果跳过出错的分配继续执行，随后的 `wake_up` 会报显存不足：

```
RuntimeError: CUDA Error: out of memory at /workspace/vllm/csrc/cumem_allocator.cpp:187
```

`cpp:274` 处是 ROCm 分支里 `cuMemUnmap` / `cuMemRelease`（即 `hipMemUnmap` / `hipMemRelease`）的第一个错误码，`cpp:187` 是 wake 时的 `cuMemMap`。当前代码只保留第一个错误，因此还不知道失败的是 unmap 还是 release。

## 1.2 复现条件

vLLM 引擎参数与 verl rollout 一致：Qwen3-30B-A3B，dummy 权重，TP=2，`enable_sleep_mode`，`gpu_memory_utilization=0.4`，`max_num_seqs=256`，`cudagraph_mode=FULL_AND_PIECEWISE`。MXFP8 时通过 `hf_overrides` 给出 compressed-tensors MXFP8 配置。

## 1.3 对照实验

复现脚本 `vllm_sleep_repro.py`（本目录）：不经过 verl，直接用上面的参数起 `vllm.LLM`，先生成一次，再做两轮 `sleep(level)` / `wake_up()` / 生成。用 `instrument_cumem.py` 给 `cumem.py` 的 `_put_to_sleep` 加日志：遇到失败时打印 tag、大小、是否备份到 CPU，然后继续执行，并统计成功和失败的个数。

| # | 环境 | 精度 | HIP graph | CPU 备份方式 | sleep level | 结果 |
|---|---|---|---|---|---|---|
| 1 | v26.10 + 0.34 | MXFP8 | 开 | 0.34 原样（mmap + `cudaHostRegister`） | 1 | **失败**：weights 分配 TP0 14/171、TP1 13/171 unmap 失败，wake OOM |
| 2 | v26.10 + 0.34 | BF16 | 开 | 0.34 原样 | 1 | **失败**：每个 rank 8/165 失败，wake OOM |
| 3 | v26.10 + 0.34 | MXFP8 | 关（eager） | 0.34 原样 | 1 | 通过，两轮 sleep/wake 正常 |
| 4 | v26.10 + 0.34 | BF16 | 开 | torch pinned tensor（0.32 写法） | 1 | 通过 |
| 5 | v26.10 + 0.34 | BF16 | 开 | mmap，**不**调用 `cudaHostRegister` | 1 | 通过 |
| 6 | v26.10 + 0.34 | MXFP8 | 开 | torch pinned tensor | 1 和 2 | 都通过 |
| 7 | v26.9 + 0.32 | BF16 | 开 | 0.32 原样（torch pinned） | 1 | 通过 |
| 8 | v26.10 + 0.34 + 修复 patch | MXFP8 | 开 | verl 端到端，2 步 GRPO | verl 默认 | 通过（EXIT=0，refit 正常） |

另外，不加载模型、只用 `CuMemAllocator` 分配几块显存再 sleep/wake 的最小测试（`cumem_sleep_repro.py`，256 MiB + 512 MiB，含多 chunk）在两个镜像上都通过。说明单纯的分配器 + 备份不会触发，需要真实模型和 HIP graph。

失败的分配全部是 `tag=weights`、需要备份到 CPU（offload=True）的分配，每个只有一个 chunk。大小如下：

- MXFP8：2、20、96、192 MiB。
- BF16：2、20、192、384 MiB。

其中 192 MiB（MXFP8）正好是一层 MoE w13 在 TP2 下的 FP8 权重（128×768×2048 字节），96 MiB 是 w2；BF16 的 384 / 192 MiB 分别对应 w13 / w2。2 MiB 和 20 MiB 对应哪些张量还没确认。也就是说，只有少数几层的专家权重出问题，其余权重和 KV cache 都正常。

## 1.4 0.32 → 0.34 的代码变化

`csrc/cumem_allocator.cpp` 在 0.32 和 0.34 之间完全相同。变化在 `vllm/device_allocator/cumem.py`：

| | 0.32 | 0.34 |
|---|---|---|
| CPU 备份 | 每次 sleep 新建 `torch.empty(size, pin_memory=True)` | `create_cpu_backup`：`mmap` 匿名内存，再 `cudaHostRegister` 成 pinned |
| 释放 | 交给 PyTorch | `release_cpu_backup`：`cudaHostUnregister` |
| 复用 | wake 后丢弃备份 | 备份在多轮 sleep/wake 之间复用 |
| level 2 | 直接丢弃 weights | 新增 `discardable`：只丢弃 refit 会重写的参数，其余 weights / runtime / cudagraph 照样备份 |

改动动机没有查到 PR 说明。推测是为了避开 PyTorch CachingHostAllocator 对 pinned 内存的缓存和尺寸取整，几十 GB 的备份能精确分配、及时归还，并支持多轮复用。

## 1.5 分析

**已确认：** 同时满足以下三个条件才会出错，去掉任何一个都正常（见 1.3 第 3、4、5 行）：

- 主机备份经过 `cudaHostRegister`（ROCm 上即 `hipHostRegister`）注册；
- 抓取了 HIP graph；
- 平台是 ROCm（v26.10，HIP 7.18）。

0.34 在 CUDA 上用的是同一份代码。

**未确认：**

- HIP 运行时内部的原因。一个猜测是：这块 VMM 显存既被 HIP graph 引用，又和注册过的主机内存之间做过 `hipMemcpy`，运行时对它的状态记录出错，于是拒绝 unmap/release。没有验证。
- 是 HIP 7.18 的回归，还是 ROCm 上一直存在、只是 vLLM 0.34 才第一次触发。还没有测过"vLLM 0.34 + HIP 7.17"。
- 为什么只有少数几块专家权重失败，其余 150 多块正常。
- ROCm 分支在 HIP ≥ 7.2 时会给 VMM 映射额外加一个主机访问描述符（`hipMemLocationTypeHost`）。它是否参与了这个问题，未知。

## 1.6 当前的修复（临时）

`docker/rocm/patches/vllm-0.34-rocm-cumem-pinned-backup.patch` 只改 `cumem.py`：

- 在 ROCm 上，`create_cpu_backup` 直接返回 `torch.empty(size, dtype=uint8, device="cpu", pin_memory=PIN_MEMORY)`，`release_cpu_backup` 不做任何事，即回到 0.32 的写法；
- CUDA 保持 0.34 原样。

代价是 ROCm 上的主机内存占用又回到 CachingHostAllocator 的方式，和 0.32 一样。拷贝仍然走 pinned 内存，速度不受影响。

使用方式：和 BLOCK32 patch 一样，不写进 Dockerfile，起容器时在 `/workspace/vllm` 里 `git apply`。

## 1.7 后续实验（待做）

1. **确认是否为 HIP 运行时的回归：** 在 v26.9（HIP 7.17）上编译 vLLM 0.34，用 `vllm_sleep_repro.py` 跑 1.3 第 1 行的配置。
2. **定位失败的调用：** 给 `cumem_allocator.cpp` 的 ROCm 分支加日志，分别记录每个 chunk 的 `hipMemUnmap` 和 `hipMemRelease` 返回值。
3. **主机访问描述符：** 编一个 `num_desc=1`（去掉 host access）的变体，看问题是否消失。
4. **缩小 graph 的范围：** 分别用 `cudagraph_mode=PIECEWISE` 和 `FULL` 测试。同时把失败分配的地址对应到参数名，确认 2 / 20 MiB 是哪些张量，以及它们是否被 FULL graph 引用。
5. **纯 HIP 最小复现：** 不用 vLLM，直接用 HIP API 走一遍：`hipMemCreate` / `hipMemMap` / `hipMemSetAccess`（含 host 描述符）→ 抓一个读这块显存的 graph → `hipHostRegister` 一块 mmap 内存 → `hipMemcpy` D2H → `hipMemUnmap` / `hipMemRelease`。如果能复现，就可以直接报给 ROCm/HIP。
6. **修复的代价：** 对比 pinned 和 mmap 两种写法的主机内存峰值，以及 sleep/wake 耗时。
7. **上报：** 给 vLLM 提 issue，附复现脚本和 1.3 的对照表，同时抄送 ROCm。根据 1–5 的结果，决定是 vLLM 在 ROCm 上避开 `hostRegister`，还是修 HIP 运行时。

---

# 第二部分：BF16 专家 GEMM 改用 TE 的 CK grouped GEMM

## 2.1 背景：TE 里 BF16 MoE 专家的三条 GEMM 路径

Megatron 的 MoE 专家用 TE 的 `GroupedLinear`。BF16 时，TE 按环境变量（每次调用都会读）选择路径：

| 路径 | 怎么启用 | 特点 |
|---|---|---|
| hipBLASLt 逐专家 | 默认 | 每个专家各发一次 GEMM。32 个本地专家时，一次 fwd+bwd 约 96 次 launch |
| CK grouped（`ck_tile::GroupedGemmKernel`） | `NVTE_USE_CK_GROUPED_GEMM=1`。另外，设了 `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1` 或 CUTLASS 开关时，BF16 也会走 CK | fprop、dgrad、wgrad 各一次 launch，覆盖全部专家 |
| Triton grouped | `NVTE_USE_GROUPED_GEMM_TRITON=1` | 只在非 FP8 **且** `fuse_wgrad_accumulation=False` 时生效 |

**之前的 BF16 基线其实走的是 hipBLASLt：**

- 示例脚本原来的 BF16 分支设的是 `NVTE_USE_GROUPED_GEMM_TRITON=1`，本意是用 Triton。
- 但 Megatron-Bridge 会开 `gradient_accumulation_fusion`，所以 `fuse_wgrad_accumulation=True`，Triton 路径被跳过。
- 因此之前所有 BF16 实验（包括 bf16_aiter_v2），专家实际都跑在 hipBLASLt 逐专家上。
- 用 `gradient_accumulation_fusion=False` 强制走 Triton 的真实 RL 对比（bf16_aiter_tritongmm）显示：Triton 和 hipBLASLt 的 step 时间基本相同，这条路后来放弃了。

## 2.2 旧 TE 里 CK BF16 很慢的原因

v26.9 镜像的 TE（023639b5，2026-10-20）只要 N 是 256 的倍数，CK BF16 grouped GEMM 就用 `TileCfg_256x256x64`。Qwen3 的专家宽度全都满足这个条件。

ROCm/TransformerEngine PR #783（commit 7f2ca63，2026-11-05，"fix ck_tile/triton bf16 grouped GEMM performance"）说明了问题：这个 tile 在 ROCm ≥ 7.14 的编译器下会溢出几百个寄存器，最多慢约 10 倍。PR 改成 128x256x64 / 256x128x64 / 256x256x32（累加的情况不走 N256 分支），同时更新了 Triton 的 gfx950 tile 表（gfx950-GMM.json）。

在 v26.9 上实测（`ck_bf16_tile_check.py`，只计 GEMM kernel 时间），旧 CK 只有 61–109 TFLOP/s，hipBLASLt 逐专家是 152–215 TFLOP/s：

| 形状 | 旧 CK | hipBLASLt |
|---|---|---|
| FC1 2048→1536 | 3.61 ms | 1.51 ms |
| FC2 768→2048 | 1.92 ms | 1.06 ms |

这也解释了早先 MXFP8 实验里 BF16 ref 变慢的现象：设了 `NVTE_USE_HIPKITTENS_GROUPED_GEMM=1` 后，BF16 ref 的专家被路由到 CK。

当时尝试在 v26.9 上从源码重编 TE（023639b5 + cherry-pick #783），在 CMake configure 阶段失败，没有继续。

## 2.3 v26.10 已包含修复

rocm/primus:v26.10 的 TE 是 2.21.0.dev0+b5c00f432（2026-11-11）。git 确认 7f2ca63 是它的祖先，镜像里的 gfx950-GMM.json 也和 #783 一致，所以不需要再重编 TE。

在 v26.10 上实测：

**只计 GEMM kernel 时间**（`ck_bf16_tile_check.py`），新 tile 是 256x128x64 / 128x256x64：

| 形状 | 新 CK | hipBLASLt 逐专家 | 旧 CK（v26.9） |
|---|---|---|---|
| FC1 2048→1536 | 0.737 ms（约 440 TFLOP/s） | 1.52 ms | 3.61 ms |
| FC2 768→2048 | 0.431 ms | 1.10 ms | 1.92 ms |

**完整 fwd+bwd**（`bf16_ck_v2610_check.py`）。设置：32 个本地专家（EP4 下一个 rank）、20370 个 token、专家间 token 数不均、`fuse_wgrad_accumulation=True`（和训练时一致），取 5 次平均：

| 形状 | 原脚本设置（Triton 开关，实际走 hipBLASLt） | `NVTE_USE_CK_GROUPED_GEMM=1` | CK，`fuse_wgrad=False` |
|---|---|---|---|
| FC1 2048→1536 | 1.877 ms（几十次 `Cijk_*` launch） | **0.937 ms**（3 次 `ck_tile::GroupedGemmKernel`） | 0.950 ms |
| FC2 768→2048 | 1.254 ms | **0.580 ms**（3 次 CK launch） | 0.660 ms |

CK 和 hipBLASLt 的数值对比（`ck_bf16_verify.py`，同样的输入和权重，fp32 main_grad 累加）：

| 形状 | fprop 相对误差 | dgrad 相对误差 | wgrad 相对误差 |
|---|---|---|---|
| FC1 | 2.81e-05 | 3.69e-05 | 1.84e-07 |
| FC2 | 0 | 3.61e-05 | 1.83e-07 |

结论：在 v26.10 上，BF16 专家用 CK grouped 比 hipBLASLt 逐专家快约 2 倍，数值一致，并且在 `fuse_wgrad_accumulation=True` 时 fprop、dgrad、wgrad 都走 CK。

## 2.4 脚本修改

`examples/mxfp8/run_qwen3_30b_a3b_megatron_rocm.sh` 的 BF16 分支（2026-11-24，未提交，原文件备份为 `~/qat-work/run_qwen3_30b_a3b_megatron_rocm.sh.before_ck`）：

```bash
# 原来
export NVTE_USE_GROUPED_GEMM_TRITON=1
# 现在
export NVTE_USE_CK_GROUPED_GEMM=1
```

注意事项：

- 这个设置依赖含 #783 的 TE。在 v26.9 的旧 TE 上设它，BF16 会更慢（见 2.2）。
- 改了之后，BF16 基线比以前快，所以新的 MXFP8 对 BF16 结果不能和旧的 BF16 数据（bf16_aiter_v2 等）直接比较。
- MXFP8 分支不受影响：专家仍走 HipKittens 的 MXFP8 grouped GEMM（`NVTE_USE_HIPKITTENS_GROUPED_GEMM=1`）；ref 现在也是 MXFP8，不再有 BF16 专家。

## 2.5 端到端结果

2026-11-24 20:11–22:29，在 n06 GPU 0–3 上依次跑了 `mxfp8_v2610_aiter`（W&B eijtjy54）和 `bf16ck_v2610_aiter`（W&B ix7vgepx），各 10 步，都是 EXIT=0。设置：两组都开 AITER，SEED=42，W&B 项目 `wei-cai/qwen3-30b-a3b_dapo_mi355`，两个 run 的 10 步都已同步。MXFP8 的 actor 和 ref 都是 MXFP8，没有退回 rocRoller（0 个 `.s` 文件）。

第 2–10 步各阶段耗时（秒），每格为"均值 / 中位数"：

| 阶段 | MXFP8 | BF16（CK grouped） | MXFP8 相对 BF16（均值） |
|---|---|---|---|
| step | 359.4 / 360.0 | 360.4 / 354.3 | −0.3% |
| gen | 151.0 / 150.3 | 162.7 / 158.3 | −7.2% |
| old_log_prob | 40.6 / 40.5 | 39.2 / 38.9 | +3.6% |
| ref | 41.4 / 41.2 | 41.1 / 38.5 | +0.5% |
| update_actor | 120.4 / 119.5 | 114.7 / 115.6 | +4.9% |
| update_weights | 5.7 / 5.4 | 2.2 / 2.2 | +3.5 s |
| 吞吐（tok/s/GPU） | 2279 / 2300 | 2272 / 2307 | +0.3% |

几点观察：

- **整步持平：** MXFP8 只在生成阶段占优（快约 7%）。训练前向（old_log_prob、ref）和 update_actor 比 BF16 慢 0–5%；refit 时要把权重量化成 MXFP8，update_weights 多约 3.5 s。
- **BF16 第 7 步 gen 偏高（206 s）：** 是单步波动。去掉这一步，BF16 的 gen 均值约 157 s，step 均值约 356 s，MXFP8 在整步上依然没有优势。
- **KL：** MXFP8 的 rollout_corr/kl 约 0.0044–0.0056，BF16 约 0.0017–0.0021，和之前的实验一致。
- **不要和旧数据混比：** BF16 现在是 CK grouped，镜像也换成了 v26.10 + vLLM 0.34，所以不应和旧镜像上的 BF16 结果（bf16_aiter_v2 等）直接比较。

MXFP8 日志末尾有一段 W&B atexit teardown 的 Traceback，发生在训练结束之后，不影响结果。

---

# 相关文件

本目录：

| 文件 | 用途 |
|---|---|
| `vllm_sleep_repro.py` | 第一部分的复现脚本，参数为 `{mxfp8\|bf16} [sleep_level] [eager]` |
| `instrument_cumem.py` | 给 `cumem.py` 加失败日志 |
| `pinned_backup_patch.py` / `noreg_backup_patch.py` | 1.3 第 4、5 行实验用的改法 |
| `cumem_sleep_repro.py` | 不带模型的最小分配器测试 |
| `bf16_ck_v2610_check.py` | 第二部分 fwd+bwd 的 kernel 和耗时检查 |
| `ck_bf16_verify.py` | CK 和 hipBLASLt 的数值对比 |

repo 内：

- `docker/rocm/patches/vllm-0.34-rocm-cumem-pinned-backup.patch`：第一部分的修复。
- `docker/rocm/Dockerfile.rocm.primus-v26.10`：新镜像。

n06 上（`/mnt/disk/verl/weicai12-qat/`）：

- `bench/cumem_dbg*.log` 和 `bench/cumem_dbg_*_case.log`：第一部分各组实验的原始日志。
- `bench/cumem_dbg.sh`、`cumem_dbg2.sh`：批量跑实验的脚本。
- `bench/ck_bf16_tile_check.py`、`bench/bf16_gmm_kernel_check.py`：第二部分的其他检查脚本。
- `exp/v2610_bf16_ck_kernel_check.out`：2.3 中 fwd+bwd 和数值对比的原始输出。
- `exp/v2610_smoke_mxfp8/train.log`：verl 冒烟测试打上修复后通过的那次日志。最初失败那次的日志已被覆盖，失败时的报错见 1.1。
