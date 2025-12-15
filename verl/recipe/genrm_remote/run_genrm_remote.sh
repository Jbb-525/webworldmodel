# vllm server
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve verl-team/GenRM-CI-Test-1.5B --served_model_name genrm-demo

# sglang server
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang_router.launch_server --model-path verl-team/GenRM-CI-Test-1.5B --dp-size 4

set -x
BASE_PATH="/scratch/bj2414/WebAgent"
export WANDB_API_KEY="108760f3e4154d69eada7e2253d6e696dbe34f5c"
export WANDB_PROJECT="Web-sft"

export CUDA_VISIBLE_DEVICES=0,1 

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$BASE_PATH/data/grpo_train.parquet \
    data.val_files=$BASE_PATH/data/grpo_test.parquet \
    data.train_batch_size=1 \
    data.max_prompt_length=5000 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path= binnnnnid/temp_model\
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=recipe/genrm_remote/reward_function.py \
    custom_reward_function.name=compute_score_batch \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_wma_grpo' \
    trainer.experiment_name='verl_wma_grpo' \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 \
    trainer.resume_mode='disable'
