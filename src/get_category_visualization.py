import matplotlib.pyplot as plt
import numpy as np

import json


if __name__ == "__main__":
    data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca2-7B/llama2/w_chat_template/sys_msg_v0/ppl_0.0_10.0"
    output_path = f"{data_dir}/filtered_cat_distr.png"
    data_path_list = [
        f"{data_dir}/demo_v0/filtered_ifd_0.4_1.2_statistics.json",
        f"{data_dir}/demo_v1/filtered_ifd_0.4_0.6_statistics.json",
        f"{data_dir}/demo_v2/filtered_ifd_0.6_0.8_statistics.json",
        f"{data_dir}/demo_v3/filtered_ifd_0.8_1.0_statistics.json",
        f"{data_dir}/demo_v4/filtered_ifd_1.0_1.2_statistics.json",
    ]
    n = len(data_path_list)
    
    category_map = {
        "Endangering National Security": "ENS",
        "Insulting Behavior": "IB",
        "Discriminatory Behavior": "DB",
        "Endangering Public Health": "EPH",
        "Copyright Issues": "CI",
        "Violence": "V",
        "Drugs": "D",
        "Privacy Violation": "PV",
        "Economic Crime": "EC",
        "Mental Manipulation": "MM",
        "Human Trafficking": "HT",
        "Physical Harm": "PhyH",
        "Sexual Content": "SC",
        "Cybercrime": "C",
        "Disrupting Public Order": "DPO",
        "Environmental Damage": "ED",
        "Psychological Harm": "PsyH",
        "White-Collar Crime": "WCC",
        "Animal Abuse": "AA"
    }
    
    category_list = []
    count_list = []
    label_list = ["demo_v0", "demo_v1", "demo_v2", "demo_v3", "demo_v4"]
    
    for i in range(len(data_path_list)):
        with open(data_path_list[i], "r") as f:
            data = json.load(f)
            
        if i == 0:
            category_list = list(data.keys())
            category_list.remove("summary")
        
        count_list.append(
            [
                data[category]["num_samples"]
                for category in category_list
            ]
        )
        
    category_list = [category_map[category] for category in category_list]
        
    bar_width = 0.8 / n
    
    index = np.arange(len(category_list))
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n):
        ax.bar(index + (i - (n - 1) / 2) * bar_width, count_list[i], bar_width, label=label_list[i])

    ax.set_xlabel("Categories")
    ax.set_ylabel("Count")
    ax.set_title("Category Distribution")

    ax.set_xticks(index)
    ax.set_xticklabels(category_list)

    ax.legend()

    plt.tight_layout()
    
    plt.savefig(output_path, format="png")
    
    plt.show()
    
    pass
