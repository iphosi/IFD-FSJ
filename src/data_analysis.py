import os
import json
import torch
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_perplexity_and_embedding_whole_text, get_perplexity_and_embedding_part_text

from prompts import SYSTEM_MESSAGE_V0, INSTRUCTION_PREFIX_V0, OUTPUT_PREFIX_V0


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

PROMPT_DICT_NONE = {
    "prompt_input": (
        "{instruction}\n{input}\n"
    ),
    "prompt_no_input": (
        "{instruction}\n"
    ),
}

system_message_dict = {
    "v0": {"system_message": SYSTEM_MESSAGE_V0, "instruction_prefix": INSTRUCTION_PREFIX_V0, "output_prefix": OUTPUT_PREFIX_V0},
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/alpaca_data/alpaca_data.json")
    parser.add_argument("--instruction_col_name", type=str, default="instruction")
    parser.add_argument("--input_col_name", type=str, default="input")
    parser.add_argument("--response_col_name", type=str, default="output")
    parser.add_argument("--save_path", type=str, default="debug.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="none", help="none")
    parser.add_argument("--system_message_version", type=str, default="v0")
    parser.add_argument("--analyse_instruction", action="store_true")
    parser.add_argument("--window_size", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=4)
    
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir="../cache", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir="../cache")

    model.eval()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file

    with open(args.save_path, "r") as file:
        exsisting_num =  sum(1 for _ in file)
    sampled_data = sampled_data[exsisting_num:]


    if args.prompt == "none":
        prompt_no_input = PROMPT_DICT_NONE["prompt_no_input"]
        prompt_input = PROMPT_DICT_NONE["prompt_input"]
    
    print("=" * 100)
    print(system_message_dict[args.system_message_version])
    print("=" * 100)
    print(sampled_data[0][args.instruction_col_name])
    
    if not args.analyse_instruction:
        print("=" * 100)
        print(sampled_data[0][args.response_col_name])

    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]
        instruct_i = data_i[args.instruction_col_name]
        if not args.analyse_instruction:
            output_i = data_i[args.response_col_name]
            assert len(output_i) != 0
        else:
            output_i = ""

        input_i = data_i[args.input_col_name] if args.input_col_name in data_i.keys() else ""
        if input_i == "":
            temp_dict = {"instruction":instruct_i}
            promt_to_use = prompt_no_input.format_map(temp_dict)
            whole_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": promt_to_use},
                    {"role": "assistant", "content": output_i},
                ],
                tokenize=False,
                add_generation_prompt=False
            )
            instruct_i = promt_to_use

        else:
            temp_dict = {"instruction":instruct_i,"input":input_i}
            promt_to_use = prompt_input.format_map(temp_dict)
            whole_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": promt_to_use},
                    {"role": "assistant", "content": output_i},
                ],
                tokenize=False,
                add_generation_prompt=False
            )
            instruct_i = promt_to_use
        
        if i == 0:
            print("=" * 100)
            print(whole_text)
            print("=" * 100)
            print(output_i)

        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        instruct_i_len = instruct_i_input_ids.shape[1] 

        if not args.analyse_instruction:
            ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, output_i, args.max_length-instruct_i_len+1)
            ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, args.max_length)

            temp_data_i = {}
            temp_data_i["ppl"] = [0,ppl_out_alone,0,ppl_out_condition]
            temp_data_i["loss"] = [0,loss_out_alone,0,loss_out_condition]
        else:
            ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, instruct_i, instruct_i_len+1, args.window_size, args.stride)
            
            temp_data_i = {}
            temp_data_i["ppl"] = [0,ppl_out_alone,0,0]
            temp_data_i["loss"] = [0,loss_out_alone,0,0]

        with open(args.save_path, "a") as file:
            file.write(json.dumps(temp_data_i) + "\n")
    
    print("Done: Data Analysis:",args.data_path)

if __name__ == "__main__":
    main()