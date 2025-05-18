# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py


import re
from typing import Dict, Tuple, Optional
from collections import defaultdict
import torch


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def compute_score(solution_str, ground_truth) -> float:
    retval = -2.0
    format_reward = 1
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            format_reward = 1
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 2.0
            else:
                retval = -1.5
        else:
            format_reward = -1
    except Exception as e:
        print(e)

    return retval, format_reward
class LengthRScorer:
    def __init__(self,data,tokenizer,config):
        self.data = data
        self.id2len = defaultdict(list)
        self.id2maxlen = {}
        self.id2minlen = {}
        self.tokenizer=tokenizer
        self.config=config
        
        bsz=len(data)
        index = data.non_tensor_batch['uid']
        self.index = index
        acc=0
        for i in range(bsz):
            
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
            solution_str = sequences_str
            try:
                string_in_last_boxed = last_boxed_only_string(solution_str)
                if string_in_last_boxed is not None:
                    answer = remove_boxed(string_in_last_boxed)
                    if is_equiv(answer, ground_truth):
                        acc+=1
                        if self.index[i] not in self.id2maxlen:
                            self.id2maxlen[self.index[i]] = valid_response_length
                            self.id2minlen[self.index[i]] = valid_response_length
                        else:
                            if valid_response_length > self.id2maxlen[self.index[i]]:
                                self.id2maxlen[self.index[i]] = valid_response_length
                            if valid_response_length < self.id2minlen[self.index[i]]:
                                self.id2minlen[self.index[i]] = valid_response_length
                       
            except Exception as e:
                print(e)

            self.id2len[self.index[i]].append(valid_response_length)
        global MaxACC
        acc=acc/bsz
        self.acc=acc
        if acc>MaxACC:
            MaxACC=acc
        print(f"LengthRScorer acc:{acc},MaxACC:{MaxACC}")
        
    def __call__(self, 
                 index: int,
                solution_str: str, 
                ground_truth: Dict[str, str],
                format_reward: int = 1,
                answer_reward: float = 1.0) :
        return self.compute_score(index,solution_str,ground_truth,format_reward,answer_reward)

    def compute_score(self,
                    index: int,
                    solution_str: str, 
                    ground_truth: Dict[str, str],
                    format_reward: int = 1,
                    answer_reward: float = 1.0) :
        """Computes comprehensive score for model response.
        
        Args:
            solution_str: Raw model response string
            ground_truth: Dictionary containing ground truth data
            format_reward: Points awarded/deducted for format correctness
            answer_reward: Points awarded/deducted for answer correctness
            
        Returns:
            Total score (sum of format and answer rewards)
        """

        
        answer_score, format_score = compute_score(solution_str, ground_truth)
        
        
        
        if answer_score > 0:
            
            data_item = self.data[index]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            current_len = valid_response_length
            max_len = self.id2maxlen[self.index[index]]
            min_len = self.id2minlen[self.index[index]]
            if current_len > min_len+self.config.algorithm.len_tolerance:
                # length penalty same to kimi 1.5
                length_penalty = 0.5 - (current_len - min_len) / (max_len - min_len+1e-5)
            else:
                length_penalty = 0.5
        else:
            length_penalty = 0
        
        len_reward = length_penalty 

        if self.acc >= MaxACC-self.config.algorithm.acc_tolerance:
            total_score = format_score + answer_score + len_reward

        else:
            total_score = format_score + answer_score
        
        

        return total_score

