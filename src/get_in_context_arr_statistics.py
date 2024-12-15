import pandas as pd
import json

import numpy as np

import os


if __name__ == "__main__":
    wrt_adv_prefix = True
    data_dir = "IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench-V5/llama2/w_chat_template/sys_msg_v0/harmful_behaviors_subset/ifsj_rs/demonstrations/demo_v5.1.1"
    num_shots_list = [8]

    if wrt_adv_prefix:
        output_path = f"{data_dir}/in_context_adv_arr_statistics.jsonl"
    else:
        output_path = f"{data_dir}/in_context_arr_statistics.jsonl"

    for num_shots in num_shots_list:
        if wrt_adv_prefix:
            data_path = f"{data_dir}/in_context_adv_arr_s{num_shots}.jsonl"
        else:
            data_path = f"{data_dir}/in_context_arr_s{num_shots}.jsonl"
            
        print(data_path)
            
        assert os.path.exists(data_path)

        df = pd.read_json(data_path, lines=True)
        
        assert len(df) == 50
        
        total_value_list = []
        
        if wrt_adv_prefix:
            shot_value_dict = {i: [] for i in range(num_shots)}
        else:
            shot_value_dict = {i: [] for i in range(num_shots - 1)}

        for i in range(len(df)):
            sample_in_context_ifd_arr = np.array(df[f"in_context_ifd"].iloc[i])
            
            total_value = 1 - sample_in_context_ifd_arr[0][-1] / sample_in_context_ifd_arr[-1][-1]
            
            total_value_list.append(total_value)
            
            shot_value_arr = 1 - sample_in_context_ifd_arr[:, -1][:-1] / sample_in_context_ifd_arr[:, -1][1:]
            
            for i in range(shot_value_arr.shape[0]):
                shot_value_dict[i].append(shot_value_arr[i])
                
        for i, value_list in shot_value_dict.items():
            value_arr = np.array(value_list)
            mean_value = value_arr.mean()
            std_value = value_arr.std()
            frac_pos_value = (value_arr > 0).mean()
            
            shot_value_dict[i] = {
                "mean_value": round(mean_value, 3),
                "std_value": round(std_value, 3),
                "frac_pos_value": round(frac_pos_value, 3)
            }
            
        total_value_arr = np.array(total_value_list)
        mean_total_value = total_value_arr.mean()
        std_total_value = total_value_arr.std()
        frac_pos_total_value = (total_value_arr > 0).mean()

        output_dict = {
            "num_shots": num_shots,
            "mean_total_value": round(mean_total_value, 3),
            "std_total_value": round(std_total_value, 3),
            "frac_pos_total_value": round(frac_pos_total_value, 3),
            "shot_value": shot_value_dict,
        }
        
        with open(output_path, "a") as f:
            f.write(json.dumps(output_dict) + "\n")
