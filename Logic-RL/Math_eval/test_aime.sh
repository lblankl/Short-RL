#0.78
# model1="/volume/ailab4sci/txie/ydl/short_ablation2/ShortRL-logic-standard-1e-6/actor/global_step_1686"
# CUDA_VISIBLE_DEVICES=0 python test_aime.py --model_path $model1 > ./log/aime/standard 2>&1 &


# model2="/volume/ailab4sci/txie/ydl/short_ablation2/ShortRL-kk_kimi/actor/global_step_1686"
# CUDA_VISIBLE_DEVICES=1 python test_aime.py --model_path $model2 > ./log/aime/kimi 2>&1 &

# model2="/volume/ailab4sci/txie/ydl/short_ablation2/CosFn/actor/global_step_1686"
# CUDA_VISIBLE_DEVICES=0 python test_aime.py --model_path $model2 > ./log/aime/CosFn 2>&1 &

model2="/volume/ailab4sci/txie/ydl/short_ablation2/Efficient2/actor/global_step_1686"
CUDA_VISIBLE_DEVICES=0 python test_aime.py --model_path $model2 > ./log/aime/Efficient 2>&1 &