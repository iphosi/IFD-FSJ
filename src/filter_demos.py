import pandas as pd


data_path = "Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-DeepSeek-Distill-Llama-t50/deepseek-distill-llama/w_chat_template/sys_msg_v0/unfiltered.json"
output_path = "Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-DeepSeek-Distill-Qwen-t50/deepseek-distill-qwen/w_chat_template/sys_msg_v0/demo_v5.8.8/filtered.json"

df = pd.read_json(data_path)

cols = df.columns

df["temp_id"] = [i for i in range(len(df))]

print(len(df))
print(len(df["id"].unique()))

# df = df[df["ppl_A_condition"] <= 9.0]
print(len(df))

sample_df = df.groupby("id").sample(n=1, random_state=42)
df = df[~df["temp_id"].isin(set(sample_df["temp_id"]))]

df = df.sample(n=1024 - len(sample_df), random_state=42)
sample_df = pd.concat([sample_df, df])

print(len(sample_df))
print(len(sample_df["id"].unique()))

sample_df = sample_df[cols]

sample_df.to_json(
    output_path,
    mode="w",
    lines=False,
    force_ascii=False,
    orient="records",
    indent=4
)

pass
