model=/mnt/danlongyuan/ShortR1/records/out/logicQwen2.5-7BlengthrG-ep2/actor/global_step_1250 #model path
config="vllm"
num_limit=100
max_token=8192
ntrain=0
split="test"
log_path="/mnt/danlongyuan/ShortR1/Logic-RL/eval_kk/logicQwen2.5-7BlengthrG-ep2"

mkdir -p ${log_path}

for eval_nppl in 2 3 4 5 6 7 8; do
    log_file="${log_path}/${eval_nppl}.log"
    echo "Starting job for eval_nppl: $eval_nppl, logging to $log_file"

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python /mnt/danlongyuan/ShortR1/Logic-RL/eval_kk/main_eval_instruct.py --batch_size 8 --model ${model} --max_token ${max_token} \
    --ntrain ${ntrain} --config ${config} --limit ${num_limit} --split ${split} --temperature 1.0  --top_p 1.0 \
    --problem_type "clean" --eval_nppl ${eval_nppl} 
done &  