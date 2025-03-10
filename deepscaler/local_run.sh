#!/bin/bash
# Set XFormers backend to avoid CUDA errors
# export VLLM_ATTENTION_BACKEND=XFORMERS
# Run 8K context length training
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/train/local_run_deepscaler_1.5b_8k.sh --model $MODEL_PATH