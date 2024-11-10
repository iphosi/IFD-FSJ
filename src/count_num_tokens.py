import pandas as pd
from transformers import AutoTokenizer


def count_num_tokens(text):
    tokenized_text = tokenizer(text)
    
    return len(tokenized_text["input_ids"])


data_path = "/home/dq/repos/IFD-FSJ/datasets/demonstrations/I-FSJ/mistral_demonstration_list_official.json"
model_path = "/home/dq/repos/IFD-FSJ/models/Meta-Llama-3-8B-Instruct"

df = pd.read_json(data_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

df["num_instruction_tokens"] = df["instruction"].apply(count_num_tokens)
df["num_output_tokens"] = df["output"].apply(count_num_tokens)

max_num_tokens = (df["num_instruction_tokens"] + df["num_output_tokens"]).max()

print(max_num_tokens)

pass