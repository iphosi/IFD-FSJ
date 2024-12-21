import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


def ablation_0():
    output_path = "Self-Instruct-FSJ/figures/ablation_0.png"
    data_dir = "Self-Instruct-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench-V5/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset/greedy_adv/generations/sys_msg_v0/wo_adv_prefix"
    data_path_list = [
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_16/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_32/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_64/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_128/llm_based_p1_asr_r16.jsonl",
    ]
    
    asr_list = []
    num_shots_list = [2, 4, 8]
    
    for data_path in data_path_list:
        df = pd.read_json(data_path, lines=True)
        
        asr_list.append((df["r_lvl_ars"] * 100).tolist())
        
    labels = ["16", "32", "64", "128"]

    plt.figure(figsize=(12, 10))
    plt.ylim(0, 20)

    for i, line in enumerate(asr_list):
        plt.plot(num_shots_list, line, marker="o", label=labels[i])

    plt.title("Llama-2-7b-chat-hf")
    plt.xlabel("Number of shots")
    plt.ylabel("ASR (%)")

    plt.legend(title="Number of candidates", loc="lower right")

    plt.grid(True)
    plt.show()
    
    plt.savefig(output_path, format="png")
    

def ablation_1():
    output_path = "Self-Instruct-FSJ/figures/ablation_1.png"
    data_dir = "Self-Instruct-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench-V5/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset/greedy_adv/generations/sys_msg_v0/wo_adv_prefix"
    data_path_list = [
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_16/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_32/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_64/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_128/llm_based_p1_asr_r16.jsonl",
    ]
    
    asr_list = []
    num_shots_list = [2, 4, 8]
    
    for data_path in data_path_list:
        df = pd.read_json(data_path, lines=True)
        
        asr_list.append((df["s_lvl_ars"] * 100).tolist())
        
    labels = ["16", "32", "64", "128"]

    plt.figure(figsize=(12, 10))

    for i, line in enumerate(asr_list):
        plt.plot(num_shots_list, line, marker="o", label=labels[i])

    plt.title("Llama-2-7b-chat-hf")
    plt.xlabel("Number of shots")
    plt.ylabel("Sample Level ASR (%)")

    plt.legend(title="Number of candidates", loc="lower right")

    plt.grid(True)
    plt.show()
    
    plt.savefig(output_path, format="png")
    
    
def ablation_2():
    output_path = "Self-Instruct-FSJ/figures/ablation_2.png"
    data_dir = "Self-Instruct-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench-V5/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset/greedy_adv/generations/sys_msg_v0/wo_adv_prefix"
    data_path_list = [
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_64/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_0.8_cand_64/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_1.0_cand_64/llm_based_p1_asr_r16.jsonl",
    ]
    
    asr_list = []
    num_shots_list = [2, 4, 8]
    
    for data_path in data_path_list:
        df = pd.read_json(data_path, lines=True)
        
        asr_list.append((df["r_lvl_ars"] * 100).tolist())
        
    labels = ["0.6", "0.8", "1.0"]
    
    plt.figure(figsize=(12, 10))
    
    for i, line in enumerate(asr_list):
        plt.plot(num_shots_list, line, marker="o", label=labels[i])

    plt.title("Llama-2-7b-chat-hf")
    plt.xlabel("Number of shots")
    plt.ylabel("ASR (%)")

    plt.legend(title="Similarity Threshold", loc="lower right")

    plt.grid(True)
    plt.show()
    
    plt.savefig(output_path, format="png")
    
    
def ablation_3():
    output_path = "Self-Instruct-FSJ/figures/ablation_3.png"
    data_dir = "Self-Instruct-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench-V5/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset/greedy_adv/generations/sys_msg_v0/wo_adv_prefix"
    data_path_list = [
        f"{data_dir}/demo_v5.1.1_sim_0.6_cand_64/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_0.8_cand_64/llm_based_p1_asr_r16.jsonl",
        f"{data_dir}/demo_v5.1.1_sim_1.0_cand_64/llm_based_p1_asr_r16.jsonl",
    ]
    
    asr_list = []
    num_shots_list = [2, 4, 8]
    
    for data_path in data_path_list:
        df = pd.read_json(data_path, lines=True)
        
        asr_list.append((df["s_lvl_ars"] * 100).tolist())
        
    labels = ["0.6", "0.8", "1.0"]
    
    plt.figure(figsize=(12, 10))
    
    for i, line in enumerate(asr_list):
        plt.plot(num_shots_list, line, marker="o", label=labels[i])

    plt.title("Llama-2-7b-chat-hf")
    plt.xlabel("Number of shots")
    plt.ylabel("Sample Level ASR (%)")

    plt.legend(title="Similarity Threshold", loc="lower right")

    plt.grid(True)
    plt.show()
    
    plt.savefig(output_path, format="png")


if __name__ == "__main__":
    ablation_3()