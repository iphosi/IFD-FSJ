import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from statistics import mean

import pandas as pd
import json

from tqdm import tqdm


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="IFD-FSJ/models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=4096)
    
    parser.add_argument("--demo_version", type=str, default="demo_v0")
    parser.add_argument("--demo_path", type=str, default="IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0/filtered_llama2_ifd_0.0_0.4_fla_577_256.json")
    
    parser.add_argument("--data_dir", type=str, default="IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors/w_adv_prompt")
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--num_responses_per_instruction", type=int, default=4)

    args = parser.parse_args()
    
    return args


def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad(): 
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()

    except:
        return 0, 0


def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        end_token = input_ids.shape[1]

        labels = input_ids.clone()
        labels[0, :start_token] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()
    
    except:
        return 0, 0


def get_in_context_ifd(tokenizer, model, max_length, instruction_list, response_list):
    conversation_list = []
    
    for i in range(len(instruction_list)):
        conversation_list.extend(
            [
                {"role": "user", "content": instruction_list[i]},
                {"role": "assistant", "content": response_list[i]},
            ]
        )
        
    in_context_ifd_matrix = [[0] * len(instruction_list) for _ in range(len(instruction_list))]
        
    for start in range(len(instruction_list)):
        for end in range(start, len(instruction_list)):
            conversation = [{"role": "system", "content": ""}] + conversation_list[2 * start:2 * (end + 1)]
            in_context_ifd_matrix[start][end] = conversation
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
            
            in_context_ifd_matrix[start][end] = in_context_ifd
            
    return in_context_ifd_matrix


def main():
    args = parse_args()
    print(args)
    
    gen_path = f"{args.data_dir}/{args.demo_version}/generation_s{args.num_shots}_r{args.num_responses_per_instruction}.json"
    output_path = f"{args.data_dir}/{args.demo_version}/in_context_ifd_s{args.num_shots}_r{args.num_responses_per_instruction}.jsonl"
    
    if os.path.exists(output_path):
        exit()
    
    demo_df = pd.read_json(args.demo_path)
    gen_df = pd.read_json(gen_path)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    for sample_idx in tqdm(range(len(gen_df))):
        sample = gen_df.iloc[sample_idx].to_dict()

        demo_idx_list = sample["demo_idx"]
        demos = demo_df.iloc[demo_idx_list]
        demo_instruction_list = demos["instruction"].tolist()
        demo_response_list = demos["output"].tolist()
        demo_ifd_list = demos["ifd_ppl"].tolist()

        conversation_list = [
            {"role": "system", "content": ""},
        ]

        in_context_ifd_matrix = get_in_context_ifd(
            tokenizer,
            model,
            args.max_length,
            demo_instruction_list,
            demo_response_list
        )
        
        output_dict = {
            "demo_idx": demo_idx_list,
            "orig_ifd": demo_ifd_list,
            "in_context_ifd": in_context_ifd_matrix
        }
        
        with open(output_path, "a") as f:
            f.write(json.dumps(output_dict) + "\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()