import pandas as pd


if __name__ == "__main__":
    data_dir = "Self-Instruct-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_ifd_0.0_0.2/greedy_adv/demonstrations/demo_v6"
    data_path_list = [
        f"{data_dir}/demo_s8.json",
        f"{data_dir}/demo_s6.json",
        f"{data_dir}/demo_s4.json",
        f"{data_dir}/demo_s2.json"
    ]
    output_path = f"{data_dir}/demo_s20.json"
    
    demo_list = None
    
    for data_path in data_path_list:
        df = pd.read_json(data_path)
    
        if not demo_list:
            demo_list = df["demo_idx"].tolist()
        else:
            for i in range(len(demo_list)):
                demo_list[i].extend(df.iloc[i]["demo_idx"])
                
    output_dict = {"demo_idx": demo_list}
    
    output_df = pd.DataFrame(output_dict)
    
    output_df.to_json(
        output_path,
        mode="w",
        lines=False,
        force_ascii=False,
        orient="records",
        indent=4
    )
            
    pass