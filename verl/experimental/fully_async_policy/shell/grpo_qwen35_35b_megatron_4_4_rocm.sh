#!/usr/bin/env bash
# Qwen3.5-35B-A3B MoE GRPO with Megatron + Fully Async Policy
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
exp_name=${EXP_NAME:-"qwen3_5_35b_megatron_fully_async_$(date +%Y%m%d_%H%M)"}

MODEL_PATH=${MODEL_PATH:-Qwen3.5-35B-A3B}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-${HOME}/data/geo3k/train.parquet}
TEST_FILE=${TEST_FILE:-${HOME}/data/geo3k/test.parquet}

rollout_mode=${ROLLOUT_MODE:-async}
rollout_name=${ROLLOUT_NAME:-vllm}   # vllm | sglang
return_raw_chat=False
if [ "${rollout_mode}" = "async" ]; then
    return_raw_chat=True
    if [ "${rollout_name}" = "vllm" ]; then
        export VLLM_USE_V1=1
    fi
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
train_ppo_micro_batch_size_per_gpu=1
infer_ppo_micro_batch_size_per_gpu=1
USE_MBRIDGE=True

# Single node with 8 GPUs. fully_async disaggregates rollout vs train onto
# SEPARATE GPUs that run concurrently, so the 8 GPUs are split:
#   4 GPUs -> rollout pool, 4 GPUs -> train pool   (N_GPUS_ROLLOUT + N_GPUS_TRAIN = 8)
NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-4}
N_GPUS_TRAIN=${N_GPUS_TRAIN:-4}

# Rollout (generation) parallelism -- runs on N_GPUS_ROLLOUT GPUs
gen_tp=${GEN_TP:-4}

# Train parallelism for Qwen3.5-35B-A3B -- runs on N_GPUS_TRAIN GPUs
# (run_qwen3_5_35b_megatron.sh uses TP=2 PP=1 EP=8 on all 8 GPUs; here EP<=4
#  because training only owns 4 GPUs)
train_tp=${TRAIN_TP:-2}
train_ep=${TRAIN_EP:-4}
train_pp=${TRAIN_PP:-1}

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=5
train_prompt_mini_bsz=32
total_rollout_steps=25600
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=False

save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-5}
total_epochs=${TOTAL_EPOCHS:-10}
resume_mode=${RESUME_MODE:-disable}
logger=${LOGGER:-"['console']"}

required_samples=$((train_prompt_mini_bsz * require_batches))
num_train_steps=${NUM_TRAIN_STEPS:-$((total_rollout_steps / (required_samples * trigger_parameter_sync_step)))}
lr_decay_steps=${LR_DECAY_STEPS:-${num_train_steps}}

# Environment
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1

mkdir -p "${CKPTS_DIR}"

# All config overrides use '++' (add-or-override)
python3 -m verl.experimental.fully_async_policy.fully_async_main \
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
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.total_training_steps=${num_train_steps} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
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
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=False \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=0 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=TRITON_ATTN \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True \
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
    actor_rollout_ref.hybrid_engine=False \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    trainer.critic_warmup=0 \
    trainer.logger="${logger}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAIN}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${resume_mode}" \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${num_train_steps} \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    "$@"
