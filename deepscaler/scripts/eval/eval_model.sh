set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="Qwen/Qwen2.5-Math-7B"
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime" "amc" "math" "minerva" "olympiad_bench")

OUTPUT_DIR=/mnt/danlongyuan/ShortR1/deepscaler/scripts/eval/out/normal  # Add default output directory
DataPath=/mnt/danlongyuan/ShortR1/deepscaler/deepscaler/data/orzmath


# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        data.path=$DataPath/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        data.n_samples=16 \
        data.batch_size=512 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.prompt_length=1024 \
        rollout.response_length=3000 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1 \
        rollout.validate_roll_out_max_length=4096
done
