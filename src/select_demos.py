import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import pandas as pd
from collections import deque

from utils import get_perplexity_and_embedding_whole_text, get_perplexity_and_embedding_part_text
from get_in_context_ifd import get_in_context_ifd_arr_row

from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--selector_mode", type=str, default="random", choices=["random", "ifd_rejection"])
    parser.add_argument("--compute_adv_prompt_ifd", action="store_true")
    
    parser.add_argument("--benchmark_name", type=str, default="AdvBench/harmful_behaviors")
    parser.add_argument("--benchmark_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench//w_chat_template/sys_msg_v0/harmful_behaviors_llama2_ifd.json")
    parser.add_argument("--benchmark_embed_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench/instruction_embed_arr.npy")
    
    parser.add_argument("--demo_version", type=str, default="demo_v3")
    parser.add_argument("--demo_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.8_1.0_fla_8642_256.json")
    parser.add_argument("--demo_embed_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.8_1.0_fla_8642_256_instruction_embed_arr.npy")
    
    parser.add_argument("--output_dir", type=str, default="IFD-FSJ/evaluation")
    
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="IFD-FSJ/models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=4096)
    
    parser.add_argument("--num_shots", type=int, default=2)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--ifd_drop_threshold", type=float, default=0.005)
    parser.add_argument("--max_num_attempts", type=int, default=8)
    
    parser.add_argument("--system_message", type=str, default="")
    
    args = parser.parse_args()
    
    return args


class DemoSelector:
    def __init__(
        self,
        selector_mode,
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
        compute_adv_prompt_ifd=False,
        in_context_ifd_path=None,
        tokenizer=None,
        model=None,
        max_length=4096
    ):
        self.compute_adv_prompt_ifd = compute_adv_prompt_ifd
        self.in_context_ifd_path = in_context_ifd_path
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length


def ifd_rejection_strategy(
    tokenizer,
    model,
    max_length,
    shot_idx,
    in_context_ifd_arr,
    cand_instruction,
    cand_response,
    cand_ifd,
    history_conversation_list,
    ifd_drop_threshold
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
    
    if shot_idx < len(in_context_ifd_arr) - 1:
        ifd_drop_arr = np.array(in_context_ifd_arr[shot_idx + 1]) - np.array(row)
        ifd_drop_flag = (ifd_drop_arr[shot_idx + 1:] > ifd_drop_threshold).all()
        
        if ifd_drop_flag:
            in_context_ifd_arr[shot_idx] = row
            
        return ifd_drop_flag
    else:
        in_context_ifd_arr[shot_idx] = row
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
    selector_mode,
    num_shots,
    sim_threshold,
    ifd_drop_threshold,
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
    orig_ifd_drop_threshold = ifd_drop_threshold
    assert orig_ifd_drop_threshold != 0
    
    for i in tqdm(range(len(instruction_list))):
        selected_idx_list = deque([])
        selected_ifd_list = deque([])
        history_conversation_list = deque([])
        curr_ifd_drop_threshold = orig_ifd_drop_threshold
        
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
            
        j = 0
            
        while j < num_shots:
            num_attempts = 0
            selected = False
            shot_idx = num_shots - j - 1
            
            history = set()

            while not selected and num_attempts < max_num_attempts:
                cand_idx = np.random.randint(demo_instruction_embed_arr.shape[0])
                
                if cand_idx in history:
                    continue
                
                history.add(cand_idx)
                
                sim_flag = similarity_strategy(
                    cand_idx=cand_idx,
                    instruction_embed=instruction_embed_arr[i],
                    demo_instruction_embed_arr=demo_instruction_embed_arr,
                    selected_idx_list=selected_idx_list,
                    sim_threshold=sim_threshold
                )
                
                if sim_flag:
                    if selector_mode == "random":
                        selected = True
                    elif selector_mode == "ifd_rejection":
                        selected = ifd_rejection_strategy(
                            tokenizer=tokenizer,
                            model=model,
                            max_length=max_length,
                            shot_idx=shot_idx,
                            in_context_ifd_arr=in_context_ifd_arr,
                            cand_instruction=demo_instruction_list[cand_idx],
                            cand_response=demo_response_list[cand_idx],
                            cand_ifd=demo_ifd_list[cand_idx],
                            history_conversation_list=history_conversation_list,
                            ifd_drop_threshold=curr_ifd_drop_threshold
                        )
                    else:
                        raise NotImplementedError
                    
                if selected:
                    break

                num_attempts += 1
                
            if selected:
                selected_idx_list.appendleft(cand_idx)
                selected_ifd_list.appendleft(demo_ifd_list[cand_idx])
                history_conversation_list.appendleft(
                    {"role": "assistant", "content": demo_response_list[cand_idx]}
                )
                history_conversation_list.appendleft(
                    {"role": "user", "content": demo_instruction_list[cand_idx]}
                )
                j += 1
            else:
                curr_ifd_drop_threshold -= abs(orig_ifd_drop_threshold)
                
                if selected_idx_list:
                    selected_idx_list.popleft()
                    selected_ifd_list.popleft()
                    history_conversation_list.popleft()
                    history_conversation_list.popleft()
                    j -= 1
                
                print("=" * 100)
                print("Max number of attempts is reached.")
                print("Relax the ifd drop constraint and restart.")
                print(f"Current ifd drop threshold: {curr_ifd_drop_threshold}")
        
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
    

def main():
    args = parse_args()
    print(args)
    
    if args.compute_adv_prompt_ifd:
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

    shot_list = demo_selection(
        selector_mode=args.selector_mode,
        num_shots=args.num_shots,
        sim_threshold=args.sim_threshold,
        ifd_drop_threshold=args.ifd_drop_threshold,
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