import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_perplexity_and_embedding_whole_text, get_perplexity_and_embedding_part_text

from statistics import mean

import pandas as pd
import json

from tqdm import tqdm

import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="IFD-FSJ/models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=4096)
    
    parser.add_argument("--demo_version", type=str, default="demo_v0")
    parser.add_argument("--demo_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.0_0.4_fla_577_256.json")
    
    parser.add_argument("--demo_data_dir", type=str, default="IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors/random/demonstrations")
    parser.add_argument("--num_shots", type=int, default=4)
    
    parser.add_argument("--adv_prompt_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench/w_chat_template/sys_msg_v0/harmful_behaviors_llama2_ifd.json")
    parser.add_argument("--compute_adv_prompt_ifd", action="store_true")

    args = parser.parse_args()
    
    return args


def get_in_context_ifd_arr_row(tokenizer, model, max_length, history_conversation_list, top_conversation_list, top_conversation_ifd):
    in_context_ifd_arr_row = [0] * ((len(history_conversation_list) // 2) + 1)
    in_context_ifd_arr_row[0] = top_conversation_ifd
    
    for i in range(1, (len(history_conversation_list) // 2) + 1):
        conversation = [{"role": "system", "content": ""}]
        conversation.extend(top_conversation_list)
        conversation.extend(history_conversation_list[:2 * i])
        
        whole_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        instruction = tokenizer.apply_chat_template(
            conversation[:-1],
            tokenize=False,
            add_generation_prompt=False
        )
        response = conversation[-1]["content"]
        
        instruction_input_ids = tokenizer.encode(instruction, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        instruction_len = instruction_input_ids.shape[1]

        ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, response, max_length - instruction_len + 1)
        ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, response, max_length)

        in_context_ifd = ppl_out_condition / ppl_out_alone
        
        in_context_ifd_arr_row[i] = in_context_ifd
        
    return in_context_ifd_arr_row


def get_in_context_ifd_arr_col(tokenizer, model, max_length, history_conversation_list, bottom_conversation_list, bottom_conversation_ifd):
    in_context_ifd_arr_col = [0] * ((len(history_conversation_list) // 2) + 1)
    in_context_ifd_arr_col[-1] = bottom_conversation_ifd
    
    for i in range((len(history_conversation_list) // 2) - 1, -1, -1):
        conversation = [{"role": "system", "content": ""}]
        conversation.extend(history_conversation_list[2 * i:])
        conversation.extend(bottom_conversation_list)
        
        whole_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        instruction = tokenizer.apply_chat_template(
            conversation[:-1],
            tokenize=False,
            add_generation_prompt=False
        )
        response = conversation[-1]["content"]
        
        instruction_input_ids = tokenizer.encode(instruction, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        instruction_len = instruction_input_ids.shape[1]

        ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, response, max_length - instruction_len + 1)
        ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, response, max_length)

        in_context_ifd = ppl_out_condition / ppl_out_alone
        
        in_context_ifd_arr_col[i] = in_context_ifd
        
    return in_context_ifd_arr_col


def get_in_context_ifd_arr(tokenizer, model, max_length, instruction_list, response_list, ifd_list):
    in_context_ifd_arr = []
    history_conversation_list = []
    
    for i in range(len(instruction_list)):
        bottom_conversation_list = [
            {"role": "user", "content": instruction_list[i]},
            {"role": "assistant", "content": response_list[i]},
        ]
        
        col = get_in_context_ifd_arr_col(
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            history_conversation_list=history_conversation_list,
            bottom_conversation_list=bottom_conversation_list,
            bottom_conversation_ifd=ifd_list[i]
        )
        col = col + [0] * (len(instruction_list) - len(col))
        in_context_ifd_arr.append(col)
        
        history_conversation_list.extend(bottom_conversation_list)
        
    in_context_ifd_arr = np.column_stack(in_context_ifd_arr).tolist()
    
    return in_context_ifd_arr


def main():
    args = parse_args()
    print(args)
    
    demo_idx_path = f"{args.demo_data_dir}/{args.demo_version}/demo_s{args.num_shots}.json"
    
    in_context_ifd_path = f"{args.demo_data_dir}/{args.demo_version}/in_context_ifd_s{args.num_shots}.jsonl"
    in_context_ifd_adv_path = f"{args.demo_data_dir}/{args.demo_version}/in_context_ifd_adv_s{args.num_shots}.jsonl"
    
    if args.compute_adv_prompt_ifd:
        output_path = in_context_ifd_adv_path
    else:
        output_path = in_context_ifd_path

    print("=" * 100)
    print(output_path)
    
    if os.path.exists(output_path):
        exit()
        
    if os.path.exists(in_context_ifd_path):
        in_context_ifd_df = pd.read_json(in_context_ifd_path, lines=True)
        print("Load pre-computed IFD.")
    else:
        in_context_ifd_df = None
        print("Compute IFD from scratch.")
    
    demo_df = pd.read_json(args.demo_path)
    demo_idx_df = pd.read_json(demo_idx_path)
    adv_prompt_df = pd.read_json(args.adv_prompt_path)
    
    if args.compute_adv_prompt_ifd:
        assert "adv_prompt" in adv_prompt_df.columns
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    for sample_idx in tqdm(range(len(demo_idx_df))):
        demo_idx_list = demo_idx_df.iloc[sample_idx]["demo_idx"]
        demos = demo_df.iloc[demo_idx_list]
        demo_instruction_list = demos["instruction"].tolist()
        demo_response_list = demos["output"].tolist()
        demo_ifd_list = demos["ifd_ppl"].tolist()

        if in_context_ifd_df is None:
            if args.compute_adv_prompt_ifd:
                demo_instruction_list.append(adv_prompt_df.iloc[sample_idx]["instruction"])
                demo_response_list.append(adv_prompt_df.iloc[sample_idx]["adv_prompt"])
                demo_ifd_list.append(adv_prompt_df.iloc[sample_idx]["ifd_ppl"])
                
            in_context_ifd_arr = get_in_context_ifd_arr(
                tokenizer=tokenizer,
                model=model,
                max_length=args.max_length,
                instruction_list=demo_instruction_list,
                response_list=demo_response_list,
                ifd_list=demo_ifd_list
            )
        else:
            in_context_ifd_arr = in_context_ifd_df.iloc[sample_idx]["in_context_ifd"]
            
            history_conversation_list = []
            
            for i in range(len(demo_instruction_list)):
                history_conversation_list.extend(
                    [
                        {"role": "user", "content": demo_instruction_list[i]},
                        {"role": "assistant", "content": demo_response_list[i]},
                    ]
                )
            
            bottom_conversation_list = [
                {"role": "user", "content": adv_prompt_df.iloc[sample_idx]["instruction"]},
                {"role": "assistant", "content": adv_prompt_df.iloc[sample_idx]["adv_prompt"]},
            ]
            
            adv_prompt_ifd_col = get_in_context_ifd_arr_col(
                tokenizer=tokenizer,
                model=model,
                max_length=args.max_length,
                history_conversation_list=history_conversation_list,
                bottom_conversation_list=bottom_conversation_list,
                bottom_conversation_ifd=adv_prompt_df.iloc[sample_idx]["ifd_ppl"]
            )
            
            in_context_ifd_arr = np.array(in_context_ifd_arr)
            in_context_ifd_arr = np.vstack(
                [in_context_ifd_arr, np.zeros((1, in_context_ifd_arr.shape[1]))]
            )
            adv_prompt_ifd_col = np.array(adv_prompt_ifd_col).reshape(-1, 1)
            in_context_ifd_arr = np.hstack((in_context_ifd_arr, adv_prompt_ifd_col)).tolist()
        
        if args.compute_adv_prompt_ifd:
            demo_ifd_list.pop()
        
        output_dict = {
            "demo_idx": demo_idx_list,
            "orig_ifd": demo_ifd_list,
            "adv_prompt_ifd": adv_prompt_df.iloc[sample_idx]["ifd_ppl"],
            "in_context_ifd": in_context_ifd_arr
        }
        
        with open(output_path, "a") as f:
            f.write(json.dumps(output_dict) + "\n")


if __name__ == "__main__":
    main()