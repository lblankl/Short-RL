#!/bin/bash
# Set XFormers backend to avoid CUDA errors
# export VLLM_ATTENTION_BACKEND=XFORMERS
# Run 8K context length training
export MODEL_PATH="Qwen/Qwen2.5-Math-7B"
SCRIPT=$1

sleep 10 # wait for ray to start
if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
ray job submit --address="http://127.0.0.1:8265"  --runtime-env-json '{"env_vars": {"WANDB_API_KEY": "", "HF_TOKEN": ""}}'  -- bash ./scripts/train/${SCRIPT} --model $MODEL_PATH
    # bash ./scripts/train/${SCRIPT} --model $MODEL_PATH
fi