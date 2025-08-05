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

import os
import argparse
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/selfplay_data/selfplay_prompts')
    parser.add_argument('--prompt_version', default='arithmetic_v1')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_examples', type=int, default=32768)
    args = parser.parse_args()

    data_source = 'yolo/multiply-3_digit' # using multiply parser for all selfplay setups

    def make_map_fn(split):
        def process_fn(_, idx):
            if args.prompt_version == 'arithmetic_v1':
                prompt = (
                    "Generate a three-digit arithmetic problem (up to three digits). "
                    "Make sure the numbers are not similar and appear unpredictable. "
                    "Do not solve the problem."
                )
            elif args.prompt_version == 'linear_equations_v1':
                prompt = (
                    "Create three diverse, challenging algebra word problems that involve linear equations with up to two variables. "
                    "Use only integers for all coefficients. The problem should be solvable with a unique solution where each variable "
                    "has a rational (integer or fractional) value. Do not solve the problems. Then select the last one and put it in the format: "
                    "Selected Question: <question>"
                )
            elif args.prompt_version == 'coding_v1':
                prompt = (
                    "Generate an original programming problem similar in style and difficulty to LeetCode easy problems.\n\n"
                    "Requirements:\n"
                    "- The problem must take a single line of space-separated integers as input and produce either a single integer or a space-separated list of integers as output.\n"
                    "- Provide 5 test cases in the exact format: INPUT_STRING ||| OUTPUT_STRING (no explanations, no extra text, OUTPUT_STRING must include the trailing '\\n' if needed).\n\n"
                    "Example Output Format:\n"
                    "Problem Description:\n"
                    "You are given a list of integers. Write a program that reads the list and returns <expected output>.\n\n"
                    "Input:\n"
                    "A single line contains space-separated integers a_1, a_2, ..., a_n (−1000 <= a_i <= 1000).\n\n"
                    "Output:\n"
                    "Print a single integer — <expected output>.\n\n"
                    "Test Cases:\n"
                    "8 -3 7 0 2 ||| 14\n"
                    "-2 5 -4 3 ||| 2\n"
                    "10 -10 ||| 0\n"
                    "4 ||| 4\n"
                    "-5 -1 -4 ||| -10\n"
                )
            else:
                raise ValueError(f"Unsupported prompt_version: {args.prompt_version}")

            return {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": 0
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }

        return process_fn

    # Create dummy dataset
    dummy_dataset = Dataset.from_dict({"dummy": [0] * args.num_examples})
    mapped_dataset = dummy_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir + "_" + args.prompt_version
    os.makedirs(local_dir, exist_ok=True)
    mapped_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
