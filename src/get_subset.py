import pandas as pd
from transformers import AutoTokenizer


data_path = "Self-Instruct-FSJ/datasets/benchmarks/AdvBench-V0/llama2/w_chat_template/sys_msg_v0/harmful_behaviors.json"
id_path = "Self-Instruct-FSJ/datasets/benchmarks/AdvBench-V0/harmful_behaviors_subset.json"
output_path = "Self-Instruct-FSJ/datasets/benchmarks/AdvBench-V0/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset.json"

# data_path = "Self-Instruct-FSJ/datasets/benchmarks/HarmBench-V6/llama3/w_chat_template/sys_msg_v0/harmbench_behaviors.json"
# id_path = "Self-Instruct-FSJ/datasets/benchmarks/HarmBench-V0/harmbench_behaviors_subset.json"
# output_path = "Self-Instruct-FSJ/datasets/benchmarks/HarmBench-V6/llama3/w_chat_template/sys_msg_v0/harmbench_behaviors_subset.json"

df = pd.read_json(data_path)

id_list = pd.read_json(id_path)["id"].tolist()

df = df[df["id"].isin(set(id_list))].copy()

df.to_json(
    output_path,
    mode="w",
    lines=False,
    force_ascii=False,
    orient="records",
    indent=4
)

pass