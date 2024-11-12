import pandas as pd
import json

import numpy as np


if __name__ == "__main__":
    data_dir = "IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors/random/w_adv_prompt"
    demo_version = "demo_v0"
    num_responses_per_instruction = 4
    num_shots_list = [i for i in range(2, 9)]

    output_path = f"{data_dir}/{demo_version}/in_context_ifd_statistics_r{num_responses_per_instruction}.jsonl"

    for num_shots in num_shots_list:
        data_path = f"{data_dir}/{demo_version}/in_context_ifd_s{num_shots}_r{num_responses_per_instruction}.jsonl"

        df = pd.read_json(data_path, lines=True)
        
        assert len(df) == 520
        
        ifd_drop_arr = []

        for i in range(len(df)):
            sample_orig_ifd_arr = np.array(df["orig_ifd"].iloc[i])
            sample_in_context_ifd_arr = np.array(df["in_context_ifd"].iloc[i])
            
            sample_ifd_drop_arr = np.triu(
                np.vstack((sample_in_context_ifd_arr[1:], sample_in_context_ifd_arr[:1])) - sample_in_context_ifd_arr,
                k=1
            )
            sample_ifd_drop_list = []
            
            for row in sample_ifd_drop_arr[:-1]:
                if any(row != 0):
                    sample_ifd_drop_list.append(row[row != 0].sum())
                else:
                    sample_ifd_drop_list.append(0)
            
            ifd_drop_arr.append(sample_ifd_drop_list)
            
            # print(sample_in_context_ifd_arr)
            # print(np.vstack((sample_in_context_ifd_arr[1:], sample_in_context_ifd_arr[:1])))
            # print(sample_ifd_drop_arr)
            # print(sample_ifd_drop_list)
            
        ifd_drop_arr = np.array(ifd_drop_arr)

        mean_ifd_drop = ifd_drop_arr.mean()
        std_ifd_drop = ifd_drop_arr.std()
        frac_pos_ifd_drop = (ifd_drop_arr > 0).sum() / (ifd_drop_arr.shape[0] * ifd_drop_arr.shape[1])
        
        output_dict = {
            "num_shots": num_shots,
            "mean_ifd_drop": round(mean_ifd_drop, 3),
            "std_ifd_drop": round(std_ifd_drop, 3),
            "frac_pos_ifd_drop": round(frac_pos_ifd_drop, 3),
            "ifd_drop": ifd_drop_arr.tolist()
        }
        
        with open(output_path, "a") as f:
            f.write(json.dumps(output_dict) + "\n")
