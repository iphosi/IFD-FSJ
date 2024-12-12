import pandas as pd
from transformers import AutoTokenizer


data_path = "IFD-FSJ/datasets/benchmarks/AdvBench-V5/starlinglm/w_chat_template/sys_msg_v0/harmful_behaviors.json"
id_path = "IFD-FSJ/datasets/benchmarks/AdvBench-V0/harmful_behaviors_subset.json"
output_path = "IFD-FSJ/datasets/benchmarks/AdvBench-V5/starlinglm/w_chat_template/sys_msg_v0/harmful_behaviors_subset.json"

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