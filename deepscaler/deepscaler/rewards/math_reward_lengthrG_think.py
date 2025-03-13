"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd

from deepscaler.system_prompts import ORM_PROMPT
from deepscaler.utils import call_gemini_llm, call_oai_rm_llm
import re
from typing import Dict, Tuple, Optional
from collections import defaultdict
import torch
global MaxACC
MaxACC=0
ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True
    
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        print(f"Model Response: {model_response}")
        
        
        model_solution = model_response
        model_answer,processed_str = extract_solution(model_solution)
        format_correct = validate_response_structure(processed_str)
        if not format_correct:
            return -2, -1
        if model_answer is None:
            return -2,-1

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        #print("ground_truths: ", ground_truths)
        if ground_truths is None:
            return -2,-1
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        #print("processed_ground_truths: ", processed_ground_truths)
        if not processed_ground_truths:
            return -2,-1

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return 2,1

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        
        
                
        return -1.5,1

def compute_score(solution_str, ground_truth,enable_llm=False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    answer_reward,format_reward = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return answer_reward,format_reward

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
           
            answer_score, format_score = compute_score(solution_str, ground_truth)
            if answer_score > 0:
                acc+=1
                if self.index[i] not in self.id2maxlen:
                    self.id2maxlen[self.index[i]] = valid_response_length
                    self.id2minlen[self.index[i]] = valid_response_length
                else:
                    if valid_response_length > self.id2maxlen[self.index[i]]:
                        self.id2maxlen[self.index[i]] = valid_response_length
                    if valid_response_length < self.id2minlen[self.index[i]]:
                        self.id2minlen[self.index[i]] = valid_response_length
                       
            

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
            if current_len > min_len+self.config.algorithm.length_tolerance:
                # length penalty same to kimi 1.5
                length_penalty = 0.5 - (current_len - min_len) / (max_len - min_len+1e-5)
            else:
                length_penalty = 0.5
        else:
            length_penalty = 0
        print("Length penalty: ", length_penalty)
        
        len_reward = length_penalty 

        if self.acc >= MaxACC-self.config.algorithm.acc_tolerance:
            print("Apply length reward: ", len_reward)
            total_score = format_score + answer_score + len_reward

        else:
            total_score = format_score + answer_score
        
        

        return total_score


if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
