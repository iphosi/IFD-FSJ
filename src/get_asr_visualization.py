import os
import pandas as pd

import matplotlib.pyplot as plt


eval_template_version = 1
num_responses_per_instruction = 8
data_dir = "Self-Instruct-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors_subset_50/random/w_adv_prompt"
output_path = f"{data_dir}/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.png"
data_path_list = [
    f"{data_dir}/demo_v0/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl",
    f"{data_dir}/demo_v1/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl",
    f"{data_dir}/demo_v2/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl",
    f"{data_dir}/demo_v3/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl",
    f"{data_dir}/demo_v4/llm_based_p{eval_template_version}_asr_r{num_responses_per_instruction}.jsonl"
]
label_list = ["ifd_0.0_0.4", "ifd_0.4_0.6", "ifd_0.6_0.8", "ifd_0.8_1.0", "ifd_1.0_inf"]

data_list = []

for data_path in data_path_list:
    assert os.path.exists(data_path)
    df = pd.read_json(data_path, lines=True)
    
    data_list.append({"num_shots": df["num_shots"].tolist(), "r_lvl_ars": df["r_lvl_ars"].tolist()})

plt.figure(figsize=(8, 6))

for label, data in zip(label_list, data_list):
    plt.plot(data["num_shots"], data["r_lvl_ars"], label=label)

plt.title("ASR Curve")
plt.xlabel("Num Shots")
plt.ylabel("r_lvl_ars")

plt.legend()

plt.grid(True)
plt.savefig(output_path, format="png")

plt.show()
