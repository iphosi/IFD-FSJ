import pandas as pd
import json

import numpy as np


if __name__ == "__main__":
    selector_mode = "random"
    compute_adv_prompt_ifd = True
    data_dir = f"IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors_subset_50/{selector_mode}/demonstrations"
    demo_version = "demo_v3"
    num_shots_list = [i for i in range(2, 15, 2)]

    if compute_adv_prompt_ifd:
        output_path = f"{data_dir}/{demo_version}/in_context_ifd_adv_statistics.jsonl"
    else:
        output_path = f"{data_dir}/{demo_version}/in_context_ifd_statistics.jsonl"

    for num_shots in num_shots_list:
        if compute_adv_prompt_ifd:
            data_path = f"{data_dir}/{demo_version}/in_context_ifd_adv_s{num_shots}.jsonl"
        else:
            data_path = f"{data_dir}/{demo_version}/in_context_ifd_s{num_shots}.jsonl"

        df = pd.read_json(data_path, lines=True)
        
        assert len(df) == 50
        
        ifd_drop_list = []
        value_list = []
        total_value_list = []
        bottom_ifd_drop_list = []

        for i in range(len(df)):
            sample_in_context_ifd_arr = np.array(df["in_context_ifd"].iloc[i])
            
            bottom_ifd_drop = sample_in_context_ifd_arr[-1][-1] - sample_in_context_ifd_arr[0][-1]
            bottom_ifd_drop_list.append(bottom_ifd_drop)
            
            sample_ifd_drop_arr = np.triu(
                np.vstack((sample_in_context_ifd_arr[1:], sample_in_context_ifd_arr[:1])) - sample_in_context_ifd_arr,
                k=1
            )
            sample_value_arr = sample_ifd_drop_arr.sum(axis=1)[:-1]
            value_list.extend(sample_value_arr.tolist())
            total_value_list.append(sample_value_arr.sum())
            
            element_indices = np.triu_indices_from(sample_ifd_drop_arr, k=1)
            sample_ifd_drop_arr = sample_ifd_drop_arr[element_indices]
            ifd_drop_list.extend(sample_ifd_drop_arr.tolist())
            
            pass
            
        ifd_drop_arr = np.array(ifd_drop_list)
        value_arr = np.array(value_list)
        total_value_arr = np.array(total_value_list)
        bottom_ifd_drop_arr = np.array(bottom_ifd_drop_list)

        mean_ifd_drop = ifd_drop_arr.mean()
        std_ifd_drop = ifd_drop_arr.std()
        frac_pos_ifd_drop = (ifd_drop_arr > 0).sum() / ifd_drop_arr.shape[0]
        
        mean_value = value_arr.mean()
        std_value = value_arr.std()
        frac_pos_value = (value_arr > 0).sum() / value_arr.shape[0]
        
        mean_total_value = total_value_arr.mean()
        std_total_value = total_value_arr.std()
        frac_pos_total_value = (total_value_arr > 0).sum() / total_value_arr.shape[0]
        
        mean_bottom_ifd_drop = bottom_ifd_drop_arr.mean()
        std_bottom_ifd_drop = bottom_ifd_drop_arr.std()
        frac_pos_bottom_ifd_drop = (bottom_ifd_drop_arr > 0).sum() / bottom_ifd_drop_arr.shape[0]
        
        output_dict = {
            "num_shots": num_shots,
            "mean_ifd_drop": round(mean_ifd_drop, 3),
            "std_ifd_drop": round(std_ifd_drop, 3),
            "frac_pos_ifd_drop": round(frac_pos_ifd_drop, 3),
            "mean_value": round(mean_value, 3),
            "std_value": round(std_value, 3),
            "frac_pos_value": round(frac_pos_value, 3),
            "mean_total_value": round(mean_total_value, 3),
            "std_total_value": round(std_total_value, 3),
            "frac_pos_total_value": round(frac_pos_total_value, 3),
            "mean_bottom_ifd_drop": round(mean_bottom_ifd_drop, 3),
            "std_bottom_ifd_drop": round(std_bottom_ifd_drop, 3),
            "frac_pos_bottom_ifd_drop": round(frac_pos_bottom_ifd_drop, 3)
        }
        
        with open(output_path, "a") as f:
            f.write(json.dumps(output_dict) + "\n")
