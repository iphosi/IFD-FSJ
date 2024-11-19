import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import pandas as pd
from collections import deque

from get_in_context_ifd import get_in_context_ifd_arr_row
from demo_selector import DemoSelector

from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--selector_mode", type=str, default="random", choices=["random", "ifd_rejection"])
    parser.add_argument("--compute_adv_prompt_ifd", action="store_true")
    
    parser.add_argument("--benchmark_name", type=str, default="AdvBench/harmful_behaviors_subset_50")
    parser.add_argument("--benchmark_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench/w_chat_template/sys_msg_v0/harmful_behaviors_subset_50_llama2_ifd.json")
    parser.add_argument("--benchmark_embed_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench/w_chat_template/sys_msg_v0/harmful_behaviors_subset_50_llama2_ifd_instruction_embed_arr.npy")
    
    parser.add_argument("--demo_version", type=str, default="demo_v3")
    parser.add_argument("--demo_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.8_1.0_fla_8642_256.json")
    parser.add_argument("--demo_embed_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.8_1.0_fla_8642_256_instruction_embed_arr.npy")
    
    parser.add_argument("--output_dir", type=str, default="IFD-FSJ/evaluation")
    
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="IFD-FSJ/models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=4096)
    
    parser.add_argument("--num_shots", type=int, default=2)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--ifd_drop_threshold", type=float, default=0.1)
    parser.add_argument("--relax_ratio", type=float, default=0.1)
    parser.add_argument("--max_num_attempts", type=int, default=8)
    
    parser.add_argument("--system_message", type=str, default="")
    
    args = parser.parse_args()
    
    return args
    

def main():
    args = parse_args()
    print(args)
    
    if "ifd" in args.selector_mode and args.compute_adv_prompt_ifd:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.selector_mode}_adv"
    else:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.selector_mode}"
        
    demo_output_dir = f"{output_dir}/demonstrations/{args.demo_version}"
    demo_output_path = f"{demo_output_dir}/demo_s{args.num_shots}.json"
    
    if not os.path.exists(demo_output_dir):
        os.makedirs(demo_output_dir)
        
    if args.selector_mode == "random":
        tokenizer = None
        model = None
        in_context_ifd_path = None
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        if args.compute_adv_prompt_ifd:
            in_context_ifd_path = f"{demo_output_dir}/in_context_ifd_adv_s{args.num_shots}.jsonl"
        else:
            in_context_ifd_path = f"{demo_output_dir}/in_context_ifd_s{args.num_shots}.jsonl"
    
    print("=" * 100)
    print(f"Demonstration output path:\n{demo_output_path}")
    print(f"In-context ifd output path:\n{in_context_ifd_path}")
    
    df = pd.read_json(args.benchmark_path)
    demo_df = pd.read_json(args.demo_path)
    
    adv_prompt_list = df["adv_prompt"].tolist()
    adv_prompt_ifd_list = df["ifd_ppl"].tolist()
    
    instruction_list = df["instruction"].tolist()
    instruction_embed_arr = np.load(args.benchmark_embed_path)
    
    demo_instruction_list = demo_df["instruction"].tolist()
    demo_response_list = demo_df["output"].tolist()
    demo_ifd_list = demo_df["ifd_ppl"].tolist()
    demo_instruction_embed_arr = np.load(args.demo_embed_path)
    
    print("=" * 100)
    print("Select demonstrations.")
    selector = DemoSelector(
        selector_mode=args.selector_mode,
        num_shots=args.num_shots,
        sim_threshold=args.sim_threshold,
        ifd_drop_threshold=args.ifd_drop_threshold,
        relax_ratio=args.relax_ratio,
        max_num_attempts=args.max_num_attempts,
        demo_instruction_embed_arr=demo_instruction_embed_arr,
        demo_instruction_list=demo_instruction_list,
        demo_response_list=demo_response_list,
        demo_ifd_list=demo_ifd_list,
        instruction_embed_arr=instruction_embed_arr,
        instruction_list=instruction_list,
        adv_prompt_list=adv_prompt_list,
        adv_prompt_ifd_list=adv_prompt_ifd_list,
        compute_adv_prompt_ifd=args.compute_adv_prompt_ifd,
        in_context_ifd_path=in_context_ifd_path,
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length
    )
    shot_list = selector.demo_selection()

    if args.num_shots > 0:
        demo_output_dict = {"demo_idx": shot_list}
        
        demo_output_df = pd.DataFrame(demo_output_dict)
        
        demo_output_df.to_json(
            demo_output_path,
            mode="w",
            lines=False,
            force_ascii=False,
            orient="records",
            indent=4
        )
    
    with open(f"{demo_output_dir}/demo_path.json", "w") as f:
        json.dump({"demo_path": args.demo_path}, f, indent=4)


if __name__ == "__main__":
    main()