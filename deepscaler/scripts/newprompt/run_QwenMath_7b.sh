
MODEL_PATH="Qwen/Qwen2.5-7B"


export WANDB_API_KEY=
export HF_TOKEN=
basepath="./deepscaler/data/ThinksimpleRL"
# Train over a single node, 8 A100-80GB GPUs.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$basepath/train.parquet \
    data.val_files=[$basepath/aime.parquet,$basepath/amc.parquet,$basepath/math.parquet,$basepath/minerva.parquet,$basepath/olympiad_bench.parquet] \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=3000 \
    data.use_template=False \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.validate_roll_out_max_length=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='SimpleRLMath_GRPO_NewP' \
    trainer.experiment_name='SimpleRLMath_Qwen_7bep1NewP' \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.test_freq=4 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 "${@:1}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir="/mnt/danlongyuan/ShortR1/records/out/SimpleRLMath_Qwen_7bep1NewP" \
    algorithm.adv_estimator=grpo
    
