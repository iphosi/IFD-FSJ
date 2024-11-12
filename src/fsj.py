import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

import torch
import numpy as np

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import json
import pandas as pd

from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str, default="random", choices=["random", "ifd_greedy"])
    parser.add_argument("--do_generation", action="store_true")
    
    parser.add_argument("--benchmark_name", type=str, default="AdvBench/harmful_behaviors")
    parser.add_argument("--benchmark_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench/harmful_behaviors.json")
    parser.add_argument("--benchmark_embed_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench/instruction_embed_arr.npy")
    parser.add_argument("--use_adv_prompt", action="store_true")
    
    parser.add_argument("--demo_version", type=str, default="demo_v0")
    parser.add_argument("--demo_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.0_0.4_fla_577_256.json")
    parser.add_argument("--demo_embed_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.0_0.4_fla_577_256_instruction_embed_arr.npy")
    
    parser.add_argument("--output_dir", type=str, default="IFD-FSJ/evaluation")
    
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="IFD-FSJ/models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=4096)
    
    parser.add_argument("--num_shots", type=int, default=2)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--max_num_attempts", type=int, default=20)
    
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=100)
    
    parser.add_argument("--system_message", type=str, default="")
    
    args = parser.parse_args()
    
    return args


def random_demo_selection(
    num_shots,
    sim_threshold,
    max_num_attempts,
    instruction_embed_arr,
    demo_instruction_embed_arr
):
    shot_list = []
    
    for i in tqdm(range(instruction_embed_arr.shape[0])):
        selected_idx_list = []
            
        for j in range(num_shots):
            num_attempts = 0
            selected = False

            while not selected and num_attempts < max_num_attempts:
                cand_idx = np.random.randint(demo_instruction_embed_arr.shape[0])
                
                if cand_idx in selected_idx_list:
                    num_attempts += 1
                    continue
                
                sim_flag_list = [
                    (
                        torch.cosine_similarity(
                            torch.tensor(demo_instruction_embed_arr[cand_idx]),
                            torch.tensor(instruction_embed_arr[i]),
                            dim=0
                        ) <= sim_threshold
                    ).item()
                ]
                
                if not sim_flag_list[-1]:
                    num_attempts += 1
                    continue
                
                for idx in selected_idx_list:
                    sim_flag_list.append(
                        (
                            torch.cosine_similarity(
                                torch.tensor(demo_instruction_embed_arr[cand_idx]),
                                torch.tensor(demo_instruction_embed_arr[idx]),
                                dim=0
                            ) <= sim_threshold
                        ).item()
                    )
                    
                    if not sim_flag_list[-1]:
                        break
                
                if all(sim_flag_list):
                    selected = True
                    selected_idx_list.append(cand_idx)
                    break
                else:
                    num_attempts += 1
                    
        assert len(selected_idx_list) == num_shots
        
        shot_list.append(selected_idx_list)
        
    return shot_list


def ifd_greedy_demo_selection(
    
):
    pass


def fsj(
    tokenizer,
    model,
    sampling_params,
    system_message,
    instruction_list,
    shot_list,
    demo_instruction_list,
    demo_response_list,
    adv_prompt_list,
    use_adv_prompt
):
    query_list = []
    
    for i in range(len(instruction_list)):
        conversation_list = [
            {"role": "system", "content": system_message},
        ]

        for demo_idx in shot_list[i]:
            conversation_list.extend(
                [
                    {"role": "user", "content": demo_instruction_list[demo_idx]},
                    {"role": "assistant", "content": demo_response_list[demo_idx]},
                ]
            )
            
        conversation_list.append(
            {"role": "user", "content": instruction_list[i]}
        )
        
        if use_adv_prompt and adv_prompt_list:
            adv_prompt = adv_prompt_list[i]
        else:
            adv_prompt = ""
            
        query_list.append(
            tokenizer.apply_chat_template(
                conversation_list,
                tokenize=False,
                add_generation_prompt=True
            ) + adv_prompt
        )
    
    print("=" * 100)
    print(query_list[0])
    print("=" * 100)
    
    output_list = model.generate(
        query_list,
        sampling_params,
    )
    response_list = []
    
    for output in output_list:
        response_list.append(
            [
                output.outputs[i].text
                for i in range(sampling_params.n)
            ]
        )
        
    return response_list
    

def main():
    args = parse_args()
    print(args)
    
    output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.mode}"
    
    if args.use_adv_prompt:
        output_dir = f"{output_dir}/w_adv_prompt/{args.demo_version}"
    else:
        output_dir = f"{output_dir}/wo_adv_prompt/{args.demo_version}"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    demo_output_path = f"{output_dir}/demo_s{args.num_shots}_r{args.num_return_sequences}.json"
    generation_output_path = f"{output_dir}/generation_s{args.num_shots}_r{args.num_return_sequences}.json"
    
    df = pd.read_json(args.benchmark_path)
    demo_df = pd.read_json(args.demo_path)
    
    adv_prompt_list = df["adv_prompt"].tolist()
    
    instruction_list = df["instruction"].tolist()
    instruction_embed_arr = np.load(args.benchmark_embed_path)
    
    demo_instruction_list = demo_df["instruction"].tolist()
    demo_response_list = demo_df["output"].tolist()
    demo_instruction_embed_arr = np.load(args.demo_embed_path)
    
    if os.path.exists(demo_output_path):
        print("=" * 100)
        print("Load demonstrations.")
        shot_list = pd.read_json(demo_output_path)["demo_idx"].tolist()
    else:
        print("=" * 100)
        print("Select demonstrations.")

        shot_list = random_demo_selection(
            num_shots=args.num_shots,
            sim_threshold=args.sim_threshold,
            max_num_attempts=args.max_num_attempts,
            instruction_embed_arr=instruction_embed_arr,
            demo_instruction_embed_arr=demo_instruction_embed_arr
        )
        
        if args.num_shots != 0:
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
        
    if args.do_generation:
        print("=" * 100)
        print("Do few-shot jailbreaking.")
        device_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=args.max_length,
            padding_side="left",
            truncation_side="left",
        )
        
        sampling_params = SamplingParams(
            n=args.num_return_sequences,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
            
        model = LLM(
            model=args.model_path,
            max_model_len=args.max_length,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=len(device_list),
            dtype="auto",
            swap_space=8
        )
            
        response_list = fsj(
            tokenizer=tokenizer,
            model=model,
            sampling_params=sampling_params,
            system_message=args.system_message,
            instruction_list=instruction_list,
            shot_list=shot_list,
            demo_instruction_list=demo_instruction_list,
            demo_response_list=demo_response_list,
            adv_prompt_list=adv_prompt_list,
            use_adv_prompt=args.use_adv_prompt
        )
        
        df["model_response"] = response_list
        df["demo_idx"] = shot_list
        
        df.to_json(
            generation_output_path,
            mode="w",
            lines=False,
            force_ascii=False,
            orient="records",
            indent=4
        )
    
    with open(f"{output_dir}/demo_path.json", "w") as f:
        json.dump({"demo_path": args.demo_path}, f, indent=4)


if __name__ == "__main__":
    main()