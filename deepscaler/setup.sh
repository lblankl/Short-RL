

pip install -e ./verl
pip install -e .
# pip install -U vllm
pip uninstall vllm -y
# pip3 install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly # important * for speeding
pip install pynvml==12.0.0

# cd tools/vllm-0.7.3
# VLLM_USE_PRECOMPILED=1 pip install --editable .
# cd ..
pip3 install vllm==0.7.3
# cd tools/vllm-0.7.3
# pip install -e .
# VLLM_USE_PRECOMPILED=1 pip install --editable .
# pip3 install vllm==0.7.3