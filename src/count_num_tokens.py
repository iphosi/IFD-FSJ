import pandas as pd
import json

from transformers import AutoTokenizer


def count_num_tokens(text, tokenizer):
    tokenized_text = tokenizer(text)
    
    return len(tokenized_text["input_ids"])


if __name__ == "__main__":
    # model_name="Llama-2-7b-chat-hf"
    # model_name="Meta-Llama-3-8B-Instruct"
    # model_name="Meta-Llama-3.1-8B-Instruct"
    # model_name="OpenChat-3.6-8B"
    # model_name="Qwen2.5-7B-Instruct"
    model_name="Starling-LM-7B-beta"

    model_path = f"Self-Instruct-FSJ/models/{model_name}"
    data_dir = "Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-StarlingLM-t50/starlinglm/w_chat_template/sys_msg_v0/demo_v5.6.6"
    data_path = f"{data_dir}/filtered.json"
    # data_dir = "Self-Instruct-FSJ/datasets/demonstrations/I-FSJ/llama2"
    # data_path = f"{data_dir}/mistral_demonstration.json"
    
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