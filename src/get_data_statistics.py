import pandas as pd
import json
import numpy as np


if __name__ == "__main__":
    # model_type = "llama3"
    model_type = "llama2"
    
    # data_dir = "IFD-FSJ/datasets/demonstrations/I-FSJ/no_sep/wo_chat_template"
    # data_dir = "IFD-FSJ/datasets/demonstrations/I-FSJ/no_sep/w_chat_template/sys_msg_v0"
    # data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca3-8B/w_chat_template/sys_msg_v0"
    data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca2-7B/w_chat_template/sys_msg_v0"
    # data_dir = "IFD-FSJ/datasets/benchmarks/AdvBench/w_chat_template/sys_msg_v0"
    
    # dataset_name = "harmful_behaviors_subset_50"
    dataset_name = "filtered"
    
    ext = "ifd_0.0_inf_fla_26116_256"
    # ext = "ifd"
    data_path = f"{data_dir}/{dataset_name}_{model_type}_{ext}.json"
    output_path = f"{data_dir}/{dataset_name}_{model_type}_{ext}_statistics.json"
    
    df = pd.read_json(data_path)
    
    num_nans = sum(df["ifd_ppl"].isnull())
    df.dropna(inplace=True)
        
    summary_dict = {
        "num_samples": len(df) + num_nans,
        "num_nans": num_nans,
        "num_categories": "-" if "harm_category" not in df.columns else len(df.iloc[0]["harm_category"]),
        "max_ppl_A_direct": round(df["ppl_A_direct"].max(), 3),
        "min_ppl_A_direct": round(df["ppl_A_direct"].min(), 3),
        "mean_ppl_A_direct": round(df["ppl_A_direct"].mean(), 3),
        "std_ppl_A_direct": round(df["ppl_A_direct"].std(), 3),
        "max_ppl_A_condition": round(df["ppl_A_condition"].max(), 3),
        "min_ppl_A_condition": round(df["ppl_A_condition"].min(), 3),
        "mean_ppl_A_condition": round(df["ppl_A_condition"].mean(), 3),
        "std_ppl_A_condition": round(df["ppl_A_condition"].std(), 3),
        "max_ifd_ppl": round(df["ifd_ppl"].max(), 3),
        "min_ifd_ppl": round(df["ifd_ppl"].min(), 3),
        "mean_ifd_ppl": round(df["ifd_ppl"].mean(), 3),
        "std_ifd_ppl": round(df["ifd_ppl"].std(), 3),
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