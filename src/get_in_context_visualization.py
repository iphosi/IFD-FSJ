import os
import pandas as pd

import matplotlib.pyplot as plt


eval_template_version = 1
num_responses_per_instruction = 8
data_dir = "IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors_subset_50/random/demonstrations"
output_path = f"{data_dir}/in_context_ifd_adv.png"
data_path_list = [
    f"{data_dir}/demo_v0/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl",
    f"{data_dir}/demo_v1/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl",
    f"{data_dir}/demo_v2/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl",
    f"{data_dir}/demo_v3/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl",
    f"{data_dir}/demo_v4/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl"
]