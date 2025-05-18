model1="/volume/ailab4sci/txie/ydl/short_ablation2/ShortRL-kk_kimi/actor/global_step_1686"
model2="/volume/ailab4sci/txie/ydl/short_ablation2/Efficient2/actor/global_step_1686"
# CUDA_VISIBLE_DEVICES=2 python test_amc.py --model_path $model1 > ./log/amc/kimi 2>&1 &
CUDA_VISIBLE_DEVICES=3 python test_amc.py --model_path $model2 > ./log/amc/Efficient 2>&1 &


