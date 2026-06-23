#!/usr/bin/env bash
# Qwen3.5-35B-A3B MoE GRPO with Megatron + Fully Async Policy
#
# Structure mirrors:
#   verl/experimental/fully_async_policy/shell/grpo_qwen3_235b_megatron_npu.sh
# Model-specific (Qwen3.5-35B-A3B) settings taken from:
#   examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
#
# Entry point:
#   verl.experimental.fully_async_policy.fully_async_main
#   config: fully_async_ppo_megatron_trainer.yaml
#
# Qwen3.5 architecture notes:
#   Qwen3.5 uses Gated Delta Net (GDN) linear attention which currently does NOT
#   support packed sequences (THD) in Megatron-LM, so force bshd compute:
#     - actor_rollout_ref.model.use_remove_padding=False
#     - actor_rollout_ref.actor.megatron.use_remove_padding=False
#     - actor_rollout_ref.actor.use_dynamic_bsz=False

set -xeuo pipefail

project_name=${PROJECT_NAME:-'verl_grpo_qwen3_5_35b_geo3k'}
exp_name=${EXP_NAME:-'qwen3_5_35b_megatron_fully_async'}

MODEL_PATH=${MODEL_PATH:-Qwen3.5-35B-A3B}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-${HOME}/data/geo3k/train.parquet}
TEST_FILE=${TEST_FILE:-${HOME}/data/geo3k/test.parquet}

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=grpo
use_kl_in_reward=False

# Response length parameters
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 2))

# Performance Related Parameter
# GDN requires bshd: no remove_padding, no dynamic bsz
use_dynamic_bsz=False
actor_ppo_max_token_len=4096
infer_ppo_max_token_len=4096
offload=True
train_ppo_micro_batch_size_per_gpu=1
infer_ppo_micro_batch_size_per_gpu=1
USE_MBRIDGE=True

# Single node with 8 GPUs. fully_async disaggregates rollout vs train onto
# SEPARATE GPUs that run concurrently, so the 8 GPUs are split:
#   4 GPUs -> rollout pool, 4 GPUs -> train pool   (N_GPUS_ROLLOUT + N_GPUS_TRAIN = 8)

# Rollout (generation) parallelism -- runs on N_GPUS_ROLLOUT (=4) GPUs
gen_tp=4

# Train parallelism for Qwen3.5-35B-A3B -- runs on N_GPUS_TRAIN (=4) GPUs
# (run_qwen3_5_35b_megatron.sh uses TP=2 PP=1 EP=8 on all 8 GPUs; here EP<=4
#  because training only owns 4 GPUs)
train_tp=2
train_ep=4
train_pp=1

# Fully async specific parameters (single node => nnodes=1 for both pools)
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
N_GPUS_TRAIN=${N_GPUS_TRAIN:-4}

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=5
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*400)))
staleness_threshold=0.5
trigger_parameter_sync_step=1
require_batches=1
partial_rollout=True

# Environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1

mkdir -p logs "${CKPTS_DIR}"

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.hybrid_engine=False \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAIN}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=100 \
    trainer.total_epochs=10 \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    "$@" 2>&1 | tee "logs/verl_qwen3_5_35b_fully_async_$(date +%Y%m%d_%H%M).log"
