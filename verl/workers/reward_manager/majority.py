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

from collections import defaultdict
import ipdb
st = ipdb.set_trace
import torch
from collections import Counter

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register

@register("majority")
class MajorityRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        id2answer = defaultdict(lambda: defaultdict(list))
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            try: 
                if "gsm8k" in data_item.non_tensor_batch["data_source"]:
                    from verl.utils.reward_score import gsm8k
                    extracted_answer = gsm8k.extract_solution(response_str)
                elif "math" in data_item.non_tensor_batch["data_source"].lower():
                    from verl.utils.reward_score import math
                    try:
                        string_in_last_boxed = math.last_boxed_only_string(response_str)
                        if string_in_last_boxed is not None:
                            extracted_answer = math.remove_boxed(string_in_last_boxed)
                        else:
                            extracted_answer = None
                    except Exception as e:
                        print(f"Error extracting answer from {response_str}: {e}")
                        extracted_answer = None                
                elif "multiply" in data_item.non_tensor_batch["data_source"].lower():
                    from verl.utils.reward_score import multiply
                    extracted_answer = multiply.extract_solution(response_str)
                else:
                    assert False, "Only gsm8k is supported for majority reward manager"
            except Exception as e:
                print(f"Error extracting answer from {response_str}: {e}")
                extracted_answer = None
            try:
                id2answer[data_item.non_tensor_batch["uid"]]['answer'].append(extracted_answer)
                id2answer[data_item.non_tensor_batch["uid"]]['index'].append(i)
            except Exception as e:
                print(f"Error extracting answer from {response_str}: {e}")
                extracted_answer = None
        
        for id, answer in id2answer.items():
            answer_count = Counter(answer['answer'])
            majority_answer = answer_count.most_common(1)[0][0]
            rewards = [1 if (answer_val == majority_answer and majority_answer != None) else 0 for answer_val in answer['answer']]
            for i, index in enumerate(answer['index']):
                id2answer[id][f'reward_{index}'].append(rewards[i])
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            assert len(id2answer[data_item.non_tensor_batch["uid"]][f'reward_{i}']) == 1, "Only one reward is allowed for each response"
            reward = id2answer[data_item.non_tensor_batch["uid"]][f'reward_{i}'][0]

            reward_tensor[i, valid_response_length - 1] = reward

            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0

            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print("[prompt]", prompt_str)
            #     print("[response]", response_str)
            #     print("[ground_truth]", ground_truth)
            #     if isinstance(score, dict):
            #         for key, value in score.items():
            #             print(f"[{key}]", value)
            #     else:
            #         print("[score]", score)
        # st()
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor