import pandas as pd
import json

from transformers import AutoTokenizer


def count_num_tokens(text):
    tokenized_text = tokenizer(text)
    
    return len(tokenized_text["input_ids"])


data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0"
data_path = f"{data_dir}/filtered_llama2_ifd_1.0_inf_fla_1239_256.json"
model_path = "IFD-FSJ/models/Llama-2-7b-chat-hf"
output_path = f"{data_dir}/filtered_llama2_ifd_num_tokens_statistics.jsonl"

df = pd.read_json(data_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

df["num_instruction_tokens"] = df["instruction"].apply(count_num_tokens)
df["num_response_tokens"] = df["output"].apply(count_num_tokens)
df["num_tokens"] = df["num_instruction_tokens"] + df["num_response_tokens"]

output_dict = {
    "max_num_tokens": int(df["num_tokens"].max()),
    "min_num_tokens": int(df["num_tokens"].min()),
    "mean_num_tokens": round(df["num_tokens"].mean(), 3),
    "std_num_tokens": round(df["num_tokens"].std(), 3),
    "max_num_instruction_tokens": int(df["num_instruction_tokens"].max()),
    "min_num_instruction_tokens": int(df["num_instruction_tokens"].min()),
    "mean_num_instruction_tokens": round(df["num_instruction_tokens"].mean(), 3),
    "std_num_instruction_tokens": round(df["num_instruction_tokens"].std(), 3),
    "max_num_response_tokens": int(df["num_response_tokens"].max()),
    "min_num_response_tokens": int(df["num_response_tokens"].min()),
    "mean_num_response_tokens": round(df["num_response_tokens"].mean(), 3),
    "std_num_response_tokens": round(df["num_response_tokens"].std(), 3),
    "data_path": data_path,
}

with open(output_path, "a") as f:
    f.write(json.dumps(output_dict) + "\n")

pass