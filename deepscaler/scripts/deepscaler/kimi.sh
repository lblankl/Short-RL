
MODEL_PATH=/volume/ailab4sci/models/DeepSeek-R1-Distill-Qwen-1.5B


export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_API_KEY=debf45c8d5066727456db660e57400d78b751446
Name=Deepscaler-kimi
SavePath=/volume/ailab4sci/txie/ydl/Short-RL/DeepScaler/$Name
basepath="./deepscaler/data/ThinkDeepScaler"
length_tolerance=100
acc_tolerance=0.05
reward_type=kimi
# Train over a single node, 8 A100-80GB GPUs.
nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$basepath/train.parquet \
    data.val_files=[$basepath/aime.parquet,$basepath/amc.parquet,$basepath/math.parquet,$basepath/minerva.parquet,$basepath/olympiad_bench.parquet] \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.use_template=False \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.validate_roll_out_max_length=9216 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='DeepScaler' \
    trainer.experiment_name=$Name \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=300 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 "${@:1}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.default_local_dir=$SavePath \
    trainer.reward_type=$reward_type \
    algorithm.acc_tolerance=$acc_tolerance \
    algorithm.length_tolerance=$length_tolerance > /volume/ailab4sci/txie/ydl/Short-RL/deepscaler/scripts/out/$Name 2>&1 &