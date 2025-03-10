import re
from typing import Dict, Tuple, Optional
from collections import defaultdict
import torch
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

def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    # print("\n[Ground Truth Parsing]")
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            # print(f"  Found: {name} → {role}")
        else:
            
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    # print("\n[Model Answer Parsing]")
    # print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    # print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        # print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            # print(f"  Found: {name} → {role}")
        else:
            # print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
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
        
        # print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        
        print("  Tag sequence validation passed")

    return validation_passed
global MaxACC
MaxACC=0

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



            solution_text = ground_truth.get('solution_text_format', '')
            gt_status = parse_solution_text_format(solution_text)
            expected_names = list(gt_status.keys())
         

            # Extract model answer
            answer_text, processed_str = extract_solution(solution_str)
         

            # Validate response structure
            format_correct = validate_response_structure(processed_str)
            if format_correct and answer_text:
                pred_status = parse_model_answer(answer_text, expected_names)
                if pred_status:
                    if pred_status == gt_status:
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
        
        

        


        

        # print("\n" + "="*80)
        # print(" Processing New Sample ".center(80, '='))
        
        # Parse ground truth data
        solution_text = ground_truth.get('solution_text_format', '')
        gt_status = parse_solution_text_format(solution_text)
        expected_names = list(gt_status.keys())
        # print(f"[Ground Truth] Final identities: {gt_status}")

        # Extract model answer
        answer_text, processed_str = extract_solution(solution_str)
        # print(f"\n[Model Response]\n{processed_str}")

        # Validate response structure
        format_correct = validate_response_structure(processed_str)
        format_score = format_reward if format_correct else -abs(format_reward)
        # print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
        # print(f"  Format score: {format_score}")

        # Validate answer content
        answer_score = 0
        if format_correct and answer_text:
            pred_status = parse_model_answer(answer_text, expected_names)
            if pred_status:
                # print(f"\n[Content Validation]")
                # print(f"  Expected: {gt_status}")
                # print(f"  Predicted: {pred_status}")
                
                if pred_status == gt_status:
                    answer_score = 2
                    # print("  Content validation: FULL MATCH")
                else:
                    answer_score = -1.5
                    # print("  Content validation: MISMATCH")
            else:
                answer_score = -2
                # print( "Fail to parse answer")
        else:
            answer_score = -2
            # print("\n[Content Validation] Skipped due to format errors or missing answer")

        # len reward = length_penalty if answer_score > 0 else min(0, length_penalty)
        
        
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
            # print(f"  Length penalty: {length_penalty}")
        else:
            total_score = format_score + answer_score
        
        # print("\n" + "-"*80)
        # print(f" Final Score ".center(80, '-'))
        # print(f"  Format: {format_score}")
        # print(f"  Answer: {answer_score}")
        # print(f"  Total: {total_score}")
        # print("="*80 + "\n")

        return total_score
