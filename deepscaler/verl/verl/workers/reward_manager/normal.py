# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from verl.utils.reward_score import math_lengthrG
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


from deepscaler.rewards.math_reward import deepscaler_reward_fn
from deepscaler.rewards import math_reward_lengthrG

def _select_rm_score_fn(data_source,config,data,tokenizer):
    if config.trainer.reward_type == 'default':
        if data_source == 'openai/gsm8k':
            return gsm8k.compute_score
        elif data_source == 'lighteval/MATH':
            return math.compute_score
        else:
            return deepscaler_reward_fn
        
    # elif config.trainer.reward_type == 'math_kimi':
    #     if data_source == 'lighteval/MATH':
    #         return math_kimi.KimiScorer(data)
    # elif config.trainer.reward_type == 'math_lengthr':
    #     if data_source == 'lighteval/MATH':
    #         return math_lengthr.LengthRScorer(data,tokenizer)
    elif config.trainer.reward_type == 'LengthrG':
        if data_source == 'lighteval/MATH':
            return math_lengthrG.LengthRScorer(data,tokenizer,config)
        else:
            return math_reward_lengthrG.LengthRScorer(data,tokenizer,config)
        
    else:
        raise NotImplementedError

class RewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score,config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.config=config

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        data_source = data[0].non_tensor_batch['data_source']
        compute_score_fn = _select_rm_score_fn(data_source,self.config,data,tokenizer=self.tokenizer)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']
            if 'kimi' in self.config.trainer.reward_type or 'Lengthr' in self.config.trainer.reward_type:
                score = compute_score_fn(index=i, solution_str=sequences_str, ground_truth=ground_truth)
            else:
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

            

            
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor

