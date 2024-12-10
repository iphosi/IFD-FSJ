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
    parser.add_argument("--demo_path", type=str, default="")
    
    parser.add_argument("--demo_data_dir", type=str, default="IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench-V1/harmful_behaviors_subset_50/random/demonstrations")
    parser.add_argument("--num_shots", type=int, default=4)
    
    parser.add_argument("--adv_prefix_path", type=str, default="IFD-FSJ/datasets/benchmarks/AdvBench-V1/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset_50.json")
    parser.add_argument("--wrt_adv_prefix", action="store_true")

    args = parser.parse_args()
    
    return args


def get_in_context_arr_row(tokenizer, model, max_length, history_conversation_list, top_conversation_list, top_conversation_ifd):
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


def get_in_context_arr_col(
    tokenizer,
    model,
    max_length,
    history_conversation_list,
    bottom_conversation_list,
    bottom_conversation_ifd,
    bottom_conversation_ppl_direct,
    bottom_conversation_ppl_condition
):
    in_context_ifd_arr_col = [0] * ((len(history_conversation_list) // 2) + 1)
    in_context_ifd_arr_col[-1] = bottom_conversation_ifd
    in_context_ppl_arr_col = [0] * ((len(history_conversation_list) // 2) + 1)
    in_context_ppl_arr_col[-1] = bottom_conversation_ppl_condition
    
    ppl_out_alone = bottom_conversation_ppl_direct
    loss_out_alone = np.log(ppl_out_alone)

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

        ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, response, max_length)

        in_context_ifd = ppl_out_condition / ppl_out_alone
        
        in_context_ifd_arr_col[i] = in_context_ifd
        in_context_ppl_arr_col[i] = ppl_out_condition
        
    return in_context_ifd_arr_col, in_context_ppl_arr_col


def get_in_context_arr_backup(tokenizer, model, max_length, instruction_list, response_list, ifd_list, ppl_direct_list, ppl_condition_list):
    in_context_ifd_arr = []
    in_context_ppl_arr = []
    history_conversation_list = []
    
    for i in range(len(instruction_list)):
        bottom_conversation_list = [
            {"role": "user", "content": instruction_list[i]},
            {"role": "assistant", "content": response_list[i]},
        ]
        
        ifd_col, ppl_col = get_in_context_arr_col(
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            history_conversation_list=history_conversation_list,
            bottom_conversation_list=bottom_conversation_list,
            bottom_conversation_ifd=ifd_list[i],
            bottom_conversation_ppl_direct=ppl_direct_list[i],
            bottom_conversation_ppl_condition=ppl_condition_list[i]
        )
        ifd_col = ifd_col + [0] * (len(instruction_list) - len(ifd_col))
        in_context_ifd_arr.append(ifd_col)
        
        ppl_col = ppl_col + [0] * (len(instruction_list) - len(ppl_col))
        in_context_ppl_arr.append(ppl_col)
        
        history_conversation_list.extend(bottom_conversation_list)
        
    in_context_ifd_arr = np.column_stack(in_context_ifd_arr).tolist()
    in_context_ppl_arr = np.column_stack(in_context_ppl_arr).tolist()
    
    return in_context_ifd_arr, in_context_ppl_arr


def get_in_context_arr(tokenizer, model, max_length, instruction_list, response_list, ifd_list):
    in_context_ifd_arr = [[0] * len(instruction_list) for _ in range(len(instruction_list))]
    
    conversation_list = []
    
    response = response_list[-1]
    
    for i in range(len(instruction_list)):
        in_context_ifd_arr[i][i] = ifd_list[i]
        
        conversation_list.extend(
            [
                {"role": "user", "content": instruction_list[i]},
                {"role": "assistant", "content": response_list[i]}
            ]
        )
    
    ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, response, max_length)
    
    for i in range(len(instruction_list) - 1):
        conversation = []
        conversation.extend(conversation_list[2 * i:])
        
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
        
        # print("=" * 100)
        # print(whole_text)
        # print("-" * 100)
        # print(instruction)
        # print("-" * 100)
        # print(response)
        # print("-" * 100)
        
        instruction_input_ids = tokenizer.encode(instruction, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        instruction_len = instruction_input_ids.shape[1]

        ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, response, max_length)

        ifd = ppl_out_condition / ppl_out_alone
        
        in_context_ifd_arr[i][-1] = ifd
        
    return in_context_ifd_arr


def main():
    args = parse_args()
    print(args)
    
    demo_idx_path = f"{args.demo_data_dir}/{args.demo_version}/demo_s{args.num_shots}.json"
    
    in_context_arr_path = f"{args.demo_data_dir}/{args.demo_version}/in_context_arr_s{args.num_shots}.jsonl"
    in_context_adv_arr_path = f"{args.demo_data_dir}/{args.demo_version}/in_context_adv_arr_s{args.num_shots}.jsonl"
    
    if args.wrt_adv_prefix:
        output_path = in_context_adv_arr_path
    else:
        output_path = in_context_arr_path

    print("=" * 100)
    print(output_path)
    
    if os.path.exists(output_path):
        exit()
        
    if os.path.exists(in_context_arr_path):
        in_context_arr_df = pd.read_json(in_context_arr_path, lines=True)
        print("Load pre-computed in-context array.")
    else:
        in_context_ifd_df = None
        print("Compute in-context array from scratch.")
    
    demo_df = pd.read_json(args.demo_path)
    demo_idx_df = pd.read_json(demo_idx_path)
    adv_prefix_df = pd.read_json(args.adv_prefix_path)
    
    if args.wrt_adv_prefix:
        assert "adv_prefix" in adv_prefix_df.columns
        
        adv_prefix_list = adv_prefix_df["adv_prefix"].tolist()
        adv_prefix_ifd_list = adv_prefix_df["ifd_ppl"].tolist()
    else:
        adv_prefix_list = None
        adv_prefix_ifd_list = None
        
    instruction_list = adv_prefix_df["instruction"].tolist()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    for sample_idx in tqdm(range(len(demo_idx_df))):
        demo_idx_list = demo_idx_df.iloc[sample_idx]["demo_idx"]
        demos = demo_df.iloc[demo_idx_list]
        demo_instruction_list = demos["instruction"].tolist()
        demo_response_list = demos["output"].tolist()
        demo_ifd_list = demos["ifd_ppl"].tolist()

        if in_context_ifd_df is None:
            if args.wrt_adv_prefix:
                demo_instruction_list.append(instruction_list[sample_idx])
                demo_response_list.append(adv_prefix_list[sample_idx])
                demo_ifd_list.append(adv_prefix_ifd_list[sample_idx])
                
            in_context_ifd_arr = get_in_context_arr(
                tokenizer=tokenizer,
                model=model,
                max_length=args.max_length,
                instruction_list=demo_instruction_list,
                response_list=demo_response_list,
                ifd_list=demo_ifd_list,
            )
            
            pass
        else:
            raise NotImplementedError
        
        output_dict = {
            "demo_idx": demo_idx_list,
            "in_context_ifd": in_context_ifd_arr,
            "orig_ifd": demo_ifd_list
        }
        
        with open(output_path, "a") as f:
            f.write(json.dumps(output_dict) + "\n")


if __name__ == "__main__":
    main()