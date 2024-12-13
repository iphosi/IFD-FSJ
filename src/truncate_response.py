import pandas as pd


data_path = "IFD-FSJ/evaluation/Starling-LM-7B-beta/AdvBench-V2/starlinglm/w_chat_template/sys_msg_v0/harmful_behaviors/random/generations/sys_msg_v0/w_adv_prefix/demo_v0/generation_s0_r128.json"
flag_path = "IFD-FSJ/evaluation/Starling-LM-7B-beta/AdvBench-V2/starlinglm/w_chat_template/sys_msg_v0/harmful_behaviors/random/generations/sys_msg_v0/w_adv_prefix/demo_v0/llm_based_p1_asr_r128.jsonl"
output_path = "IFD-FSJ/datasets/demonstrations/AdvBench-V2-StarlingLM-t50/unfiltered.json"

instruction_list = []
output_list = []
id_list = []

df = pd.read_json(data_path)
flag_list = pd.read_json(flag_path, lines=True).iloc[0]["response_flag"]

for i in range(len(df)):
    instruction = df.iloc[i]["instruction"]
    adv_prefix = df.iloc[i]["adv_prefix"]
    sample_id = df.iloc[i]["id"]
    
    for j in range(len(df.iloc[i]["model_response"])):
        flag = flag_list[i][j]
        response = df.iloc[i]["model_response"][j]
        output = adv_prefix + response
        
        end = output.rfind(". ")
        output = output[:end+1].strip()
        
        if flag and len(output) > 0:
            instruction_list.append(instruction)
            output_list.append(output)
            id_list.append(sample_id)
        
output_dict = {
    "instruction": instruction_list,
    "output": output_list,
    "id": id_list
}

output_df = pd.DataFrame(output_dict)

print(len(output_df))
print(len(output_df["id"].unique()))

output_df.to_json(
    output_path,
    mode="w",
    lines=False,
    force_ascii=False,
    orient="records",
    indent=4
)