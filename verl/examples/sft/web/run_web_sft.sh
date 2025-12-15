set -x

export PATH=$HOME/.local/bin:$PATH
BASE_PATH="/scratch/bj2414/WebAgent"
if [ "$#" -lt 1 ]; then
    echo "Usage: run_web_sft.sh <nproc_per_node> [other_configs...]"
    exit 1
fi

nproc_per_node=$1

export WANDB_API_KEY="108760f3e4154d69eada7e2253d6e696dbe34f5c"
export WANDB_PROJECT="Web-sft"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$BASE_PATH/data/sft_train.parquet \
    data.val_files=$BASE_PATH/data/sft_test.parquet \
    trainer.test_freq=100 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=8192 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=1 \
    optim.lr=5e-6 \
    optim.weight_decay=0.01 \
    model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
    trainer.default_local_dir=$BASE_PATH/checkpoints/web_sft \
    trainer.project_name=web-sft \
    trainer.experiment_name=web-sft-qwen-2.5-3b-instruct_new \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] \
    trainer.max_ckpt_to_keep=3 \
    trainer.save_freq=50 \
    model.lora_rank=32 \
    model.lora_alpha=64 \
    model.trust_remote_code=True \
    use_remove_padding=True \
    model.fsdp_config.model_dtype=bf16 \
    model.target_modules=all-linear \
    model.strategy=fsdp
