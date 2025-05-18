<div align="center">

# Short RL

**Short RL**: Efficient RL Training for Reasoning Models via
Length-Aware Optimization

<div>

</div>
</div>



<div align="center" style="line-height: 1;">
    <a href="https://github.com/lblankl/Short-RL" style="margin: 2px;"><img alt="Code" src="https://img.shields.io/badge/Short%20RL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>

<a href="https://arxiv.org/abs/submit/6449940" target="_blank">
  <img alt="Arxiv"
    src="https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white"/></a>

</div>

<div>
<br>

</div>

## Overview


Code of paper "Efficient RL Training for Reasoning Models via Length-Aware Optimization

We introduce **Short-RL**, a simple yet effective technique to control response length during the RL training process of R1-like models, while maintaining stable performance.



## Getting Started 🚀

### Installation & Training Scripts

#### Logic-RL Setup

To begin working with **Short-RL** for the Logic-RL dataset, just run:

```bash
cd Logic-RL
conda create -n logic python=3.9
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 ray
pip3 install flash-attn --no-build-isolation
pip install -e .  # For verl integration
pip install wandb IPython matplotlib
```

#### Math-RL Setup

To begin working with **Short-RL** for 3 math settings , just run:

```bash
cd deepscaler
bash setup.sh
```

#### Start Logic-RL Training

We directly use the data from logic-RL at Logic-RL/data/kk/instruct

Train Short-RL

```bash
cd Logic-RL
bash sh/Short-RL.sh # Normal-RL.sh for baseline comparision
```

The performance of Logic-RL is sensitive to the learning rate. In our experiments, a learning rate of 1e-6 with a batch size of 8 yields the best convergence within 3 epochs, which is the setting used in the paper. However, this configuration can be unstable, sometimes leading to sudden drops in test accuracy during training, regardless of whether standard RL or Short-RL is used.  For reliable reproduction of the paper results, multiple runs may be necessary.

For more stable training, a learning rate of 4e-7 is a robust alternative, though it requires more epochs to converge.

Eval

```bash
cd eval_kk
bash eval.sh

cd Math_eval
bash test_aime.sh
bash test_amc.sh
```

#### Start Math-RL Training

Data preparation:

You can directly use the data at deepscaler/data/orzmath, deepscaler/data/ThinkDeepScaler, deepscaler/data/ThinksimpleRL

Or if you want to prepare it yourself, taking Open Reasoner Zero as an example, first you need to download [`curated 57k training data from Orz`](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/data) to ./deepscaler/data.
Then run

```bash
bash ./scripts/data/data.sh
```

Train Short-RL

```bash
cd deepscaler
#Open Reasoner Zero
bash scripts/train/Short-RL.sh # Normal-RL.sh for baseline comparision
#DeepScaleR
bash scripts/deepscaler/Short-RL.sh # Normal-RL.sh for baseline comparision
#SimpleRL-Math
bash scripts/simplerl/Short-RL.sh # Normal-RL.sh for baseline comparision
```

Evaluation

The evaluation curves can be seen in wandb during training.

Or if you want to evaluate it after training. You can run:

```bash
bash ./scripts/eval/eval_model.sh
```

## Acknowledgements

Our training framework is built on [Logic-RL](https://github.com/Unakar/Logic-RL), [deepscaler](https://github.com/agentica-project/deepscaler), [verl](https://github.com/volcengine/verl) and [ray](https://github.com/ray-project/ray).

- Our model is based on [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- Our math data is from [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), [deepscaler](https://github.com/agentica-project/deepscaler),  [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)

## Citation

```bibtex
@misc{Short-RL,
  title={Short RL : Controlling the Dynamics of Training Length with better performance},
  author={Danlong Yuan, Tian Xie, Shaohan Huang, Chong Luo, Furu Wei},
  year={2025},
  howpublished={\url{https://github.com/lblankl/Short-RL}},
}
```
