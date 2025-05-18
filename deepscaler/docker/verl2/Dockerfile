FROM buaahsh/nvcr:24.07-py3-scalerl-0211

COPY ./req.txt /workspace/req.txt
RUN apt update && apt install -y vim tmux git

RUN pip install -r /workspace/req.txt

# copy folder
COPY ./vllm-0.7.3 /workspace/vllm-0.7.3

# setup environment
ENV VLLM_USE_PRECOMPILED=1
RUN cd /workspace/vllm-0.7.3 && pip install .