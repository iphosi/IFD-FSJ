import matplotlib.pyplot as plt
import numpy as np

import json


if __name__ == "__main__":
    data_dir = "IFD-FSJ/datasets/demonstrations/Alpaca3-8B/w_chat_template/sys_msg_v0"
    output_path = f"{data_dir}/filtered_llama3_ifd_cat_distr_256.png"
    data_path_list = [
        f"{data_dir}/filtered_llama3_ifd_0.0_0.4_fla_573_256_statistics.json",
        f"{data_dir}/filtered_llama3_ifd_0.4_0.6_fla_2316_256_statistics.json",
        f"{data_dir}/filtered_llama3_ifd_0.6_0.8_fla_8992_256_statistics.json",
        f"{data_dir}/filtered_llama3_ifd_0.8_1.0_fla_8077_256_statistics.json",
        f"{data_dir}/filtered_llama3_ifd_1.0_inf_fla_780_256_statistics.json",
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
    label_list = ["ifd_0.0_0.4", "ifd_0.4_0.6", "ifd_0.6_0.8", "ifd_0.8_1.0", "ifd_1.0_inf"]
    
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
