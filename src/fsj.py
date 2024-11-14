import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import json
import pandas as pd
from collections import deque

from utils import get_perplexity_and_embedding_whole_text, get_perplexity_and_embedding_part_text
from get_in_context_ifd import get_in_context_ifd_arr_row

from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fsj_mode", type=str, default="ifd_rejection", choices=["random", "ifd_rejection"])
    parser.add_argument("--compute_adv_prompt_ifd", action="store_false")
    parser.add_argument("--do_generation", action="store_true")
    
    parser.add_argument("--benchmark_name", type=str, default="AdvBench/harmful_behaviors")
    parser.add_argument("--benchmark_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench//w_chat_template/sys_msg_v0/harmful_behaviors_llama2_ifd.json")
    parser.add_argument("--benchmark_embed_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench/instruction_embed_arr.npy")
    parser.add_argument("--use_adv_prompt", action="store_true")
    
    parser.add_argument("--demo_version", type=str, default="demo_v3")
    parser.add_argument("--demo_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.8_1.0_fla_8642_256.json")
    parser.add_argument("--demo_embed_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.8_1.0_fla_8642_256_instruction_embed_arr.npy")
    
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


def get_ifd_drop(instruction, response):
    pass


def ifd_rejection_strategy(
    tokenizer,
    model,
    max_length,
    shot_idx,
    in_context_ifd_arr,
    cand_instruction,
    cand_response,
    cand_ifd,
    history_conversation_list
):
    top_conversation_list = [
        {"role": "user", "content": cand_instruction},
        {"role": "assistant", "content": cand_response}
    ]
    row = get_in_context_ifd_arr_row(
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,
        history_conversation_list=list(history_conversation_list),
        top_conversation_list=top_conversation_list,
        top_conversation_ifd=cand_ifd
    )
    row = [0] * (len(in_context_ifd_arr) - len(row)) + row
    in_context_ifd_arr[shot_idx] = row
    
    if shot_idx < len(in_context_ifd_arr) - 1:
        ifd_drop_arr = np.array(in_context_ifd_arr[shot_idx + 1]) - np.array(in_context_ifd_arr[shot_idx])
        ifd_drop_flag = (ifd_drop_arr[shot_idx + 1:] > 0).all()
        return ifd_drop_flag
    else:
        return True
    
        
def similarity_strategy(
    cand_idx,
    instruction_embed,
    demo_instruction_embed_arr,
    selected_idx_list,
    sim_threshold
):
    sim_flag = (
        torch.cosine_similarity(
            torch.tensor(demo_instruction_embed_arr[cand_idx]),
            torch.tensor(instruction_embed),
            dim=0
        ) <= sim_threshold
    ).item()
    
    if not sim_flag:
        return False
    
    for idx in selected_idx_list:
        sim_flag = (
            torch.cosine_similarity(
                torch.tensor(demo_instruction_embed_arr[cand_idx]),
                torch.tensor(demo_instruction_embed_arr[idx]),
                dim=0
            ) <= sim_threshold
        ).item()
        
        if not sim_flag:
            return False
        
    return True


def demo_selection(
    fsj_mode,
    num_shots,
    sim_threshold,
    max_num_attempts,
    demo_instruction_embed_arr,
    demo_instruction_list,
    demo_response_list,
    demo_ifd_list,
    instruction_embed_arr,
    instruction_list,
    adv_prompt_list,
    adv_prompt_ifd_list,
    compute_adv_prompt_ifd,
    in_context_ifd_path=None,
    tokenizer=None,
    model=None,
    max_length=4096
):
    shot_list = []
    
    for i in tqdm(range(len(instruction_list))):
        selected_idx_list = deque([])
        selected_ifd_list = deque([])
        history_conversation_list = deque([])
        
        if compute_adv_prompt_ifd:
            history_conversation_list.extend(
                [
                    {"role": "user", "content": instruction_list[i]},
                    {"role": "assistant", "content": adv_prompt_list[i]}
                ]
            )
            in_context_ifd_arr = [[0] * (num_shots + 1) for _ in range(num_shots + 1)]
            in_context_ifd_arr[-1][-1] = adv_prompt_ifd_list[i]
        else:
            in_context_ifd_arr = [[0] * num_shots for _ in range(num_shots)]
            
        for j in range(num_shots):
            num_attempts = 0
            selected = False

            while not selected and num_attempts < max_num_attempts:
                cand_idx = np.random.randint(demo_instruction_embed_arr.shape[0])
                
                if cand_idx in selected_idx_list:
                    num_attempts += 1
                    continue
                
                sim_flag = similarity_strategy(
                    cand_idx=cand_idx,
                    instruction_embed=instruction_embed_arr[i],
                    demo_instruction_embed_arr=demo_instruction_embed_arr,
                    selected_idx_list=selected_idx_list,
                    sim_threshold=sim_threshold
                )
                
                if sim_flag:
                    if fsj_mode == "random":
                        selected = True
                    elif fsj_mode == "ifd_rejection":
                        shot_idx = num_shots - j - 1
                        
                        selected = ifd_rejection_strategy(
                            tokenizer=tokenizer,
                            model=model,
                            max_length=max_length,
                            shot_idx=shot_idx,
                            in_context_ifd_arr=in_context_ifd_arr,
                            cand_instruction=demo_instruction_list[cand_idx],
                            cand_response=demo_response_list[cand_idx],
                            cand_ifd=demo_ifd_list[cand_idx],
                            history_conversation_list=history_conversation_list
                        )
                    else:
                        raise NotImplementedError
                    
                if selected:
                    selected_idx_list.appendleft(cand_idx)
                    selected_ifd_list.appendleft(demo_ifd_list[cand_idx])
                    history_conversation_list.appendleft(
                        {"role": "assistant", "content": demo_response_list[cand_idx]}
                    )
                    history_conversation_list.appendleft(
                        {"role": "user", "content": demo_instruction_list[cand_idx]}
                    )
                    break

                num_attempts += 1
                    
        assert len(selected_idx_list) == num_shots
        
        shot_list.append(list(selected_idx_list))
        
        if in_context_ifd_path:
            output_dict = {
                "demo_idx": list(selected_idx_list),
                "orig_ifd": list(selected_ifd_list),
                "adv_prompt_ifd": adv_prompt_ifd_list[i],
                "in_context_ifd": in_context_ifd_arr
            }
            
            with open(in_context_ifd_path, "a") as f:
                f.write(json.dumps(output_dict) + "\n")
        
    return shot_list


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
    
    device_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    
    if args.compute_adv_prompt_ifd:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.fsj_mode}_adv"
    else:
        output_dir = f"{args.output_dir}/{args.model_name}/{args.benchmark_name}/{args.fsj_mode}"
        
    demo_output_dir = f"{output_dir}/demonstrations/{args.demo_version}"
    demo_output_path = f"{demo_output_dir}/demo_s{args.num_shots}_r{args.num_return_sequences}.json"
    
    if args.use_adv_prompt:
        gen_output_dir = f"{output_dir}/w_adv_prompt/{args.demo_version}"
    else:
        gen_output_dir = f"{output_dir}/wo_adv_prompt/{args.demo_version}"
    
    gen_output_path = f"{gen_output_dir}/generation_s{args.num_shots}_r{args.num_return_sequences}.json"
    
    if not os.path.exists(demo_output_dir):
        os.makedirs(demo_output_dir)
        
    if not os.path.exists(gen_output_dir):
        os.makedirs(gen_output_dir)
        
    if args.fsj_mode == "random":
        tokenizer = None
        model = None
        in_context_ifd_path = None
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        if args.compute_adv_prompt_ifd:
            in_context_ifd_path = f"{demo_output_dir}/in_context_ifd_adv_s{args.num_shots}_r{args.num_return_sequences}.json"
        else:
            in_context_ifd_path = f"{demo_output_dir}/in_context_ifd_s{args.num_shots}_r{args.num_return_sequences}.json"
    
    
    
    
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
    
    if os.path.exists(demo_output_path):
        print("=" * 100)
        print("Load demonstrations.")
        shot_list = pd.read_json(demo_output_path)["demo_idx"].tolist()
    else:
        print("=" * 100)
        print("Select demonstrations.")

        shot_list = demo_selection(
            fsj_mode=args.fsj_mode,
            num_shots=args.num_shots,
            sim_threshold=args.sim_threshold,
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
        
    if args.do_generation and not os.path.exists(gen_output_path):
        print("=" * 100)
        print("Do few-shot jailbreaking.")
        del tokenizer, model
        
        gen_tokenizer = AutoTokenizer.from_pretrained(
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
            
        gen_model = LLM(
            model=args.model_path,
            max_model_len=args.max_length,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=len(device_list),
            dtype="auto",
            swap_space=8
        )
            
        response_list = fsj(
            tokenizer=gen_tokenizer,
            model=gen_model,
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
            gen_output_path,
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