import pandas as pd


# data_path = "IFD-FSJ/evaluation/Llama-2-7b-chat-hf/I-FSJ/llama2/w_chat_template/sys_msg_v0/mistral_demonstration/random/generations/sys_msg_v0/w_adv_prefix/demo_v0/generation_s0_r16.json"
# flag_path = "IFD-FSJ/evaluation/Llama-2-7b-chat-hf/I-FSJ/llama2/w_chat_template/sys_msg_v0/mistral_demonstration/random/generations/sys_msg_v0/w_adv_prefix/demo_v0/llm_based_p1_asr_r16.jsonl"
# output_path = "IFD-FSJ/datasets/demonstrations/I-FSJ-Llama2-t300/unfiltered.json"

data_path = "IFD-FSJ/evaluation/Meta-Llama-3-8B-Instruct/AdvBench-V2/llama3/w_chat_template/sys_msg_v0/harmful_behaviors/random/generations/sys_msg_v0/w_adv_prefix/demo_v0/generation_s0_r128.json"
flag_path = "IFD-FSJ/evaluation/Meta-Llama-3-8B-Instruct/AdvBench-V2/llama3/w_chat_template/sys_msg_v0/harmful_behaviors/random/generations/sys_msg_v0/w_adv_prefix/demo_v0/llm_based_p1_asr_r128.jsonl"
output_path = "IFD-FSJ/datasets/demonstrations/AdvBench-V2-Llama3-t50/unfiltered_.json"

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
        
        end = output.rfind(".\n")
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