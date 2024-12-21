import pandas as pd
import json
import numpy as np

import os


if __name__ == "__main__":
    data_dir = "Self-Instruct-FSJ/datasets/demonstrations/AdvBench-V5-Qwen2.5-t50/llama2/w_chat_template/sys_msg_v0/demo_v5.5.1"
    # data_dir = "Self-Instruct-FSJ/datasets/benchmarks/AdvBench-V7/llama2/w_chat_template/sys_msg_v0"
    # data_dir = "Self-Instruct-FSJ/datasets/benchmarks/I-FSJ/starlinglm/w_chat_template/sys_msg_v0"

    # dataset_name = "harmful_behaviors"
    dataset_name = "filtered"
    data_path = f"{data_dir}/{dataset_name}.json"
    output_path = f"{data_dir}/{dataset_name}_statistics.json"
    
    df = pd.read_json(data_path)
    
    num_nans = sum(df["ppl_A_condition"].isnull())
    df["ppl_A_condition"].dropna(inplace=True)
        
    summary_dict = {
        "num_samples": len(df) + num_nans,
        "num_nans": sum(df["ifd_ppl"].isnull()),
        "num_categories": "-" if "harm_category" not in df.columns else len(df.iloc[0]["harm_category"]),
        "max_ppl_A_direct": "-" if len(df) == 0 else round(float(df["ppl_A_direct"].max()), 3),
        "min_ppl_A_direct": "-" if len(df) == 0 else round(float(df["ppl_A_direct"].min()), 3),
        "mean_ppl_A_direct": "-" if len(df) == 0 else round(float(df["ppl_A_direct"].mean()), 3),
        "std_ppl_A_direct": "-" if len(df) == 0 else round(float(df["ppl_A_direct"].std()), 3),
        "max_ppl_A_condition": "-" if len(df) == 0 else round(float(df["ppl_A_condition"].max()), 3),
        "min_ppl_A_condition": "-" if len(df) == 0 else round(float(df["ppl_A_condition"].min()), 3),
        "mean_ppl_A_condition": "-" if len(df) == 0 else round(float(df["ppl_A_condition"].mean()), 3),
        "std_ppl_A_condition": "-" if len(df) == 0 else round(float(df["ppl_A_condition"].std()), 3),
        "max_ifd_ppl": "-" if len(df) == 0 else round(float(df["ifd_ppl"].max()), 3),
        "min_ifd_ppl": "-" if len(df) == 0 else round(float(df["ifd_ppl"].min()), 3),
        "mean_ifd_ppl": "-" if len(df) == 0 else round(float(df["ifd_ppl"].mean()), 3),
        "std_ifd_ppl": "-" if len(df) == 0 else round(float(df["ifd_ppl"].std()), 3),
    }
    
    if "harm_category" not in df.columns:
        statistic_dict = {}
    else:
        statistic_dict = {
            category: {
                "num_samples": 0,
                "max_ppl_A_direct": 0,
                "min_ppl_A_direct": 0,
                "mean_ppl_A_direct": [],
                "std_ppl_A_direct": 0,
                "max_ppl_A_condition": 0,
                "min_ppl_A_condition": 0,
                "mean_ppl_A_condition": [],
                "std_ppl_A_condition": 0,
                "max_ifd_ppl": 0,
                "min_ifd_ppl": 0,
                "mean_ifd_ppl": [],
                "std_ifd_ppl": 0,
            }
            for category in df.iloc[0]["harm_category"].keys()
        }
        
        for i in range(len(df)):
            for key, value in df.iloc[i]["harm_category"].items():
                if value:
                    statistic_dict[key]["num_samples"] += 1
                    statistic_dict[key]["mean_ppl_A_direct"].append(df.iloc[i]["ppl_A_direct"])
                    statistic_dict[key]["mean_ppl_A_condition"].append(df.iloc[i]["ppl_A_condition"])
                    statistic_dict[key]["mean_ifd_ppl"].append(df.iloc[i]["ifd_ppl"])
                    
        for key in statistic_dict.keys():
            if statistic_dict[key]["num_samples"] != 0:
                statistic_dict[key]["std_ppl_A_direct"] = round(np.std(statistic_dict[key]["mean_ppl_A_direct"]), 3)
                statistic_dict[key]["std_ppl_A_condition"] = round(np.std(statistic_dict[key]["mean_ppl_A_condition"]), 3)
                statistic_dict[key]["std_ifd_ppl"] = round(np.std(statistic_dict[key]["mean_ifd_ppl"]), 3)
                statistic_dict[key]["max_ppl_A_direct"] = round(np.max(statistic_dict[key]["mean_ppl_A_direct"]), 3)
                statistic_dict[key]["max_ppl_A_condition"] = round(np.max(statistic_dict[key]["mean_ppl_A_condition"]), 3)
                statistic_dict[key]["max_ifd_ppl"] = round(np.max(statistic_dict[key]["mean_ifd_ppl"]), 3)
                statistic_dict[key]["min_ppl_A_direct"] = round(np.min(statistic_dict[key]["mean_ppl_A_direct"]), 3)
                statistic_dict[key]["min_ppl_A_condition"] = round(np.min(statistic_dict[key]["mean_ppl_A_condition"]), 3)
                statistic_dict[key]["min_ifd_ppl"] = round(np.min(statistic_dict[key]["mean_ifd_ppl"]), 3)
                statistic_dict[key]["mean_ppl_A_direct"] = round(np.mean(statistic_dict[key]["mean_ppl_A_direct"]), 3)
                statistic_dict[key]["mean_ppl_A_condition"] = round(np.mean(statistic_dict[key]["mean_ppl_A_condition"]), 3)
                statistic_dict[key]["mean_ifd_ppl"] = round(np.mean(statistic_dict[key]["mean_ifd_ppl"]), 3)
            else:
                statistic_dict[key]["mean_ppl_A_direct"] = 0
                statistic_dict[key]["mean_ppl_A_condition"] = 0
                statistic_dict[key]["mean_ifd_ppl"] = 0
        
    statistic_dict["summary"] = summary_dict
        
    with open(output_path, "w") as f:
        json.dump(statistic_dict, f, indent=4)
    
    pass