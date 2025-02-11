import os

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import pandas as pd
from collections import deque

from demo_selector import DemoSelector

from tqdm import tqdm

import argparse

from prompts import (
    SYSTEM_MESSAGE_V0, INSTRUCTION_PREFIX_V0, INSTRUCTION_SUFFIX_V0
)


system_message_dict = {
    "v0": {"system_message": SYSTEM_MESSAGE_V0, "instruction_prefix": INSTRUCTION_PREFIX_V0, "instruction_suffix": INSTRUCTION_SUFFIX_V0}
}


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--selector_mode", type=str, default="greedy", choices=["random", "greedy"])
    parser.add_argument("--wrt_adv_prefix", action="store_true")
    
    parser.add_argument("--benchmark_name", type=str)
    parser.add_argument("--benchmark_path", type=str)
    parser.add_argument("--benchmark_embed_path", type=str)
    
    parser.add_argument("--demo_version", type=str)
    parser.add_argument("--demo_path", type=str)
    parser.add_argument("--demo_embed_path", type=str)
    
    parser.add_argument("--output_dir", type=str, default="Self-Instruct-FSJ/evaluation")
    
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="Self-Instruct-FSJ/models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=4096)
    
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--context_window_size", type=int, default=-1)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--lower_value_threshold", type=float, default=0.001)
    parser.add_argument("--upper_value_threshold", type=float, default=0.01)
    parser.add_argument("--relax_ratio", type=float, default=0.1)
    parser.add_argument("--num_cands_per_attempt", type=int, default=2)
    parser.add_argument("--max_num_attempts", type=int, default=8)
    
    parser.add_argument("--system_message_version", type=str, default="v0")
    
    args = parser.parse_args()
    
    return args
    

def main():
    args = parse_args()
    print(args)
    
    if args.context_window_size == -1:
        args.context_window_size = args.num_shots - 1
        window = ""
    else:
        assert args.num_shots > args.context_window_size > 0
        window = f"_w{args.context_window_size}"
    
    if args.selector_mode != "random" and args.wrt_adv_prefix:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.selector_mode}_adv"
    else:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.selector_mode}"
        
    demo_output_dir = f"{output_dir}/demonstrations/{args.demo_version}"
    demo_output_path = f"{demo_output_dir}/demo_s{args.num_shots}{window}.json"
    
    if not os.path.exists(demo_output_dir):
        os.makedirs(demo_output_dir)
        
    if args.selector_mode == "random":
        tokenizer = None
        model = None
        in_context_arr_path = None
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        if args.wrt_adv_prefix:
            in_context_arr_path = f"{demo_output_dir}/in_context_adv_arr_s{args.num_shots}{window}.jsonl"
        else:
            in_context_arr_path = f"{demo_output_dir}/in_context_arr_s{args.num_shots}{window}.jsonl"
    
    print("=" * 100)
    print(f"Demonstration output path:\n{demo_output_path}")
    print(f"In-context array output path:\n{in_context_arr_path}")
    
    if in_context_arr_path is not None and os.path.exists(in_context_arr_path):
        checkpoint = len(pd.read_json(in_context_arr_path, lines=True))
    else:
        checkpoint = 0
        
    print(f"Resume from checkpoint {checkpoint}.")
    
    df = pd.read_json(args.benchmark_path)[checkpoint:]
    demo_df = pd.read_json(args.demo_path)
    
    adv_prefix_list = df["adv_prefix"].tolist()
    adv_prefix_ifd_list = df["ifd_ppl"].tolist()
    
    instruction_list = df["instruction"].tolist()
    instruction_embed_arr = np.load(args.benchmark_embed_path)
    
    demo_instruction_list = demo_df["instruction"].tolist()
    demo_response_list = demo_df["output"].tolist()
    demo_ifd_list = demo_df["ifd_ppl"].tolist()
    demo_instruction_embed_arr = np.load(args.demo_embed_path)
    
    sample_instruction = instruction_list[0]
    sample_adv_prefix = adv_prefix_list[0]
    sample_demo_instruction = demo_instruction_list[0]
    sample_demo_response = demo_response_list[0]
    
    system_message = system_message_dict[args.system_message_version]["system_message"]
    
    print("=" * 100)
    print(f"System message:\n{system_message}")
    
    print("=" * 100)
    print(f"Sample instruction:\n{sample_instruction}")
    print("-" * 100)
    print(f"Sample adversarial prefix:\n{sample_adv_prefix}")
    print("-" * 100)
    print(f"Sample demo instruction:\n{sample_demo_instruction}")
    print("-" * 100)
    print(f"Sample demo response:\n{sample_demo_response}")
    
    print("=" * 100)
    print("Select demonstrations.")
    selector = DemoSelector(
        selector_mode=args.selector_mode,
        system_message=system_message,
        demo_instruction_embed_arr=demo_instruction_embed_arr,
        demo_instruction_list=demo_instruction_list,
        demo_response_list=demo_response_list,
        demo_ifd_list=demo_ifd_list,
        instruction_embed_arr=instruction_embed_arr,
        instruction_list=instruction_list,
        adv_prefix_list=adv_prefix_list,
        adv_prefix_ifd_list=adv_prefix_ifd_list,
        num_shots=args.num_shots,
        context_window_size=args.context_window_size,
        sim_threshold=args.sim_threshold,
        lower_value_threshold=args.lower_value_threshold,
        upper_value_threshold=args.upper_value_threshold,
        relax_ratio=args.relax_ratio,
        num_cands_per_attempt=args.num_cands_per_attempt,
        max_num_attempts=args.max_num_attempts,
        wrt_adv_prefix=args.wrt_adv_prefix,
        in_context_arr_path=in_context_arr_path,
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length
    )
    shot_list = selector.demo_selection()

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