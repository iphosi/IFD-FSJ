import pandas as pd
import json

from transformers import AutoTokenizer


def count_num_tokens(text, tokenizer):
    tokenized_text = tokenizer(text)
    
    return len(tokenized_text["input_ids"])


if __name__ == "__main__":
    model_path = "IFD-FSJ/models/Llama-2-7b-chat-hf"
    # data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v0/ppl_0.0_10.0/demo_v4"
    # data_path = f"{data_dir}/filtered_ifd_1.0_1.2.json"
    data_dir = "IFD-FSJ/datasets/demonstrations/I-FSJ/llama2_sep"
    data_path = f"{data_dir}/mistral_demonstration_list_official.json"
    
    output_path = f"{data_dir}/num_tokens_statistics.jsonl"

    df = pd.read_json(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    df["num_instruction_tokens"] = df["instruction"].apply(count_num_tokens, tokenizer=tokenizer)
    df["num_response_tokens"] = df["output"].apply(count_num_tokens, tokenizer=tokenizer)
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