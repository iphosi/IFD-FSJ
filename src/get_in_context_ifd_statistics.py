import pandas as pd
import json

import numpy as np


if __name__ == "__main__":
    data_dir = "IFD-FSJ/evaluation/Llama-2-7b-chat-hf/AdvBench/harmful_behaviors/w_adv_prompt"
    demo_version = "demo_v0"
    num_responses_per_instruction = 4
    num_shots_list = [i for i in range(2, 9)]

    output_path = f"{data_dir}/{demo_version}/in_context_ifd_statistics_r{num_responses_per_instruction}.jsonl"

    for num_shots in num_shots_list:
        data_path = f"{data_dir}/{demo_version}/in_context_ifd_s{num_shots}_r{num_responses_per_instruction}.jsonl"

        df = pd.read_json(data_path, lines=True)
        
        assert len(df) == 520
        
        ifd_gain_arr = []

        for i in range(len(df)):
            sample_orig_ifd_arr = np.array(df["orig_ifd"].iloc[i])
            sample_in_context_ifd_arr = np.array(df["in_context_ifd"].iloc[i])
            
            sample_ifd_gain_arr = np.triu(
                np.vstack((sample_in_context_ifd_arr[1:], sample_in_context_ifd_arr[:1])) - sample_in_context_ifd_arr,
                k=1
            )
            sample_ifd_gain_list = []
            
            for row in sample_ifd_gain_arr[:-1]:
                if any(row != 0):
                    sample_ifd_gain_list.append(row[row != 0].sum())
                else:
                    sample_ifd_gain_list.append(0)
            
            ifd_gain_arr.append(sample_ifd_gain_list)
            
            # print(sample_in_context_ifd_arr)
            # print(np.vstack((sample_in_context_ifd_arr[1:], sample_in_context_ifd_arr[:1])))
            # print(sample_ifd_gain_arr)
            # print(sample_ifd_gain_list)
            
        ifd_gain_arr = np.array(ifd_gain_arr)

        mean_ifd_gain = ifd_gain_arr.mean()
        std_ifd_gain = ifd_gain_arr.std()
        frac_pos_ifd_gain = (ifd_gain_arr > 0).sum() / (ifd_gain_arr.shape[0] * ifd_gain_arr.shape[1])
        
        output_dict = {
            "num_shots": num_shots,
            "mean_ifd_gain": round(mean_ifd_gain, 3),
            "std_ifd_gain": round(std_ifd_gain, 3),
            "frac_pos_ifd_gain": round(frac_pos_ifd_gain, 3),
            "ifd_gain": ifd_gain_arr.tolist()
        }
        
        with open(output_path, "a") as f:
            f.write(json.dumps(output_dict) + "\n")
