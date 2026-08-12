#!/usr/bin/env bash
set -xeuo pipefail

project_name='DAPO'
exp_name='DAPO-Qwen3.5-35B-A3B-MATH-megatron-fully-async-4-4'

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-35B-A3B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# ROCm / Megatron environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # symmetric-memory allreduce is unsupported on ROCm
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
max_model_length=$((max_prompt_length + max_response_length))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="token-mean"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
# Qwen3.5 GDN runs in BSHD layout, so sequence packing is unavailable:
# use_remove_padding=False forces use_dynamic_bsz=False and per-gpu micro batches.
use_dynamic_bsz=False
train_micro_bsz_per_gpu=1
infer_micro_bsz_per_gpu=1
actor_offload=False
ref_offload=False
gen_tp=4
train_tp=2
train_ep=4
train_pp=1
train_etp=1

# Megatron parameters
# very important! vanilla_mbridge=False needs the megatron-bridge package (module
# megatron.bridge), which is NOT in this image. `pip install megatron-bridge`, or set
# vanilla_mbridge=True to fall back to the installed standalone mbridge package.
use_mbridge=True
vanilla_mbridge=True
attn_backend=auto
moe_aux_loss_coeff=0.001 # router load-balancing loss
recompute_method=uniform
recompute_granularity=full
recompute_num_layers=1
update_weights_bucket_mb=4096

# Fully async specific parameters
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

n_gpus_rollout=4
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*100)))
test_freq=10
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=4
partial_rollout=True

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=${return_raw_chat} \
    data.trust_remote_code=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_bsz_per_gpu} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.param_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.use_mbridge=${use_mbridge} \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=${vanilla_mbridge} \
    actor_rollout_ref.actor.megatron.dtype=bfloat16 \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=${attn_backend} \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=${recompute_method} \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=${recompute_granularity} \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=${recompute_num_layers} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=${moe_aux_loss_coeff} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=${max_model_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_length} \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_mb} \
    actor_rollout_ref.ref.megatron.param_offload=${ref_offload} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.total_epochs=10 \
    trainer.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    "$@"
