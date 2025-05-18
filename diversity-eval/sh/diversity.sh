

shortrlp=/volume/ailab4sci/txie/ydl/short_ablation2/VShortRL-logic1e-6-200-1/actor
kimip=/volume/ailab4sci/txie/ydl/short_ablation2/VGShortRL-kk_kimi-0.1/actor
# export CUDA_VISIBLE_DEVICES=0
# nohup python /volume/ailab4sci/txie/ydl/Short-RL/diversity-eval/diversity_record.py --model_path $shortrlp --model_type short_rl > /volume/ailab4sci/txie/ydl/Short-RL/diversity-eval/sh/short_rl 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python /volume/ailab4sci/txie/ydl/Short-RL/diversity-eval/diversity_record.py --model_path $kimip --model_type kimi > /volume/ailab4sci/txie/ydl/Short-RL/diversity-eval/sh/kimi 2>&1 &